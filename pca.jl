#!/usr/bin/env julia
"""
PCA module (OnlinePCA.jl-backed) for omnibenchmark.

Output format (neutral HDF5, version 1) -- shared with the sibling scanpy
module. See `docs/pca_output.md` for the full spec.

File: {output_dir}/{name}_{solver}_n_{n_components}.h5

Datasets:
  /embedding       float64, shape (n_cells, n_components)
  /loadings        float64, shape (n_genes, n_components)
  /variance        float64, shape (n_components,)
  /variance_ratio  float64, shape (n_components,)
  /cell_ids        utf-8 string, shape (n_cells,)
  /gene_ids        utf-8 string, shape (n_genes,)

Root attributes:
  tool="OnlinePCA.jl", tool_version, solver,
  n_components, random_seed, format_version="1"

Implementation notes
--------------------
- Genes are centered/scaled before PCA (per-gene zero-mean unit-variance).
  OnlinePCA.jl is fed pre-scaled values with `scale="raw"` so the
  preprocessing matches the scanpy module bit-for-bit conceptually.
- OnlinePCA.jl operates on its own genes-as-rows binary format. We write
  the subset+scaled matrix to a temp binary file, run sumr, then run the
  selected solver. Outputs are reorientated to match the neutral HDF5 spec
  (cells-as-rows for /embedding, genes-as-rows for /loadings).
- Subsetting to selected genes happens here for now; this responsibility
  should move to a dedicated upstream cleanup stage.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using ArgParse
using HDF5
using SparseArrays
using Statistics
using Random
using Pkg
using GZip
using DelimitedFiles
using OnlinePCA

include(joinpath(@__DIR__, "src", "cli.jl"))
using .CLI

const OUTPUT_FORMAT_VERSION = "1"
const TOOL = "OnlinePCA.jl"

function load_selected_genes(path::AbstractString)
    if endswith(path, ".gz")
        return GZip.open(path, "r") do io
            [strip(l) for l in eachline(io) if !isempty(strip(l))]
        end
    else
        return open(path, "r") do io
            [strip(l) for l in eachline(io) if !isempty(strip(l))]
        end
    end
end

"""
Load TENx HDF5 (genes x cells) and subset to `selected_genes`.

Returns (X::SparseMatrixCSC, gene_ids::Vector{String}, cell_ids::Vector{String})
where X is genes-as-rows by cells-as-columns (OnlinePCA.jl orientation).
"""
function load_subset_matrix(h5_path::AbstractString, selected_genes::Vector{<:AbstractString})
    h5open(h5_path, "r") do h5
        g = h5["matrix"]
        data = read(g["data"])
        indices = read(g["indices"])
        indptr = read(g["indptr"])
        shape = Tuple(read(g["shape"]))  # (n_genes, n_cells)

        gene_ids = if haskey(g, "features") && haskey(g["features"], "id")
            read(g["features"]["id"])
        elseif haskey(g, "genes")
            read(g["genes"])
        else
            ["gene_$(i)" for i in 0:(shape[1]-1)]
        end

        cell_ids = haskey(g, "barcodes") ? read(g["barcodes"]) :
                   ["cell_$(i)" for i in 0:(shape[2]-1)]

        gene_ids = String.(gene_ids)
        cell_ids = String.(cell_ids)

        # TENx HDF5 stores data CSC: indptr indexes columns (cells).
        # Julia SparseMatrixCSC also uses 1-based colptr; convert.
        colptr = Vector{Int}(indptr) .+ 1
        rowval = Vector{Int}(indices) .+ 1
        nzval = Vector{Float64}(data)
        m = SparseMatrixCSC(shape[1], shape[2], colptr, rowval, nzval)

        sel_set = Set(selected_genes)
        available = Set(gene_ids)
        missing_genes = setdiff(sel_set, available)
        if !isempty(missing_genes)
            sample = sort(collect(missing_genes))[1:min(10, end)]
            error("$(length(missing_genes)) selected gene(s) not present in normalized.h5; " *
                  "first $(length(sample)): $(sample)")
        end

        mask = [gene in sel_set for gene in gene_ids]
        m_sub = m[mask, :]
        return m_sub, gene_ids[mask], cell_ids
    end
end

"""
Center and scale each row (gene) to zero mean, unit variance.
Returns a dense Float32 matrix (genes x cells).
"""
function scale_per_gene(X::SparseMatrixCSC)
    Xd = Matrix{Float32}(X)
    n_genes, n_cells = size(Xd)
    @inbounds for i in 1:n_genes
        row = @view Xd[i, :]
        mu = mean(row)
        sd = std(row; corrected = true)
        if sd == 0 || !isfinite(sd)
            row .= 0f0
        else
            row .= (row .- mu) ./ sd
        end
    end
    return Xd
end

"""
Write a (genes x cells) Float32 matrix to OnlinePCA.jl's binary format.

OnlinePCA.jl expects a JLD2 binary written via `csv2bin`, but we can
construct the same on-disk layout directly. Easier path: serialize via
`OnlinePCA.write_binary` if available; otherwise round-trip through CSV.
We use the documented `csv2bin` entrypoint for portability.
"""
function write_onlinepca_binary(X::AbstractMatrix, binfile::AbstractString)
    csvfile = binfile * ".csv"
    open(csvfile, "w") do io
        writedlm(io, X, ',')
    end
    OnlinePCA.csv2bin(csvfile = csvfile, binfile = binfile)
    rm(csvfile; force = true)
    return binfile
end

"""
Run the requested OnlinePCA.jl solver and return (eigvecs, eigvals, totalvar).

eigvecs: (n_genes, n_components)  -- gene loadings
eigvals: (n_components,)          -- variance per PC
totalvar: scalar                  -- total variance of the (scaled) matrix
"""
function run_onlinepca(binfile::AbstractString, sumrdir::AbstractString,
                       solver::AbstractString, n_components::Int, seed::Int)
    Random.seed!(seed)

    rowmean = joinpath(sumrdir, "Feature_LogMeans.csv")
    if !isfile(rowmean)
        # We pre-scaled, so any rowmean file works; sumr produces what's needed.
        rowmean = joinpath(sumrdir, "Feature_Means.csv")
    end

    common = (input = binfile,
              outdir = sumrdir,
              dim = n_components,
              scale = "raw",
              rowmeanlist = rowmean,
              logdir = sumrdir)

    out = if solver == "halko"
        OnlinePCA.halko(; common...)
    elseif solver == "ccipca"
        OnlinePCA.ccipca(; common...)
    elseif solver == "orthiter"
        OnlinePCA.orthiter(; common...)
    elseif solver == "arnoldi"
        OnlinePCA.arnoldi(; common...)
    elseif solver == "algorithm971"
        OnlinePCA.algorithm971(; common...)
    else
        error("unsupported solver: $solver")
    end

    # OnlinePCA.jl returns a tuple; the first three elements across solvers
    # are (eigenvectors, eigenvalues, totalvar) in genes-space.
    eigvecs = Matrix{Float64}(out[1])
    eigvals = Vector{Float64}(out[2])
    totalvar = Float64(out[3])
    return eigvecs, eigvals, totalvar
end

function compute_embedding(X_scaled::AbstractMatrix, loadings::AbstractMatrix)
    # X_scaled: (n_genes, n_cells); loadings: (n_genes, n_components)
    # embedding (cells, n_components) = X_scaled' * loadings
    return Matrix{Float64}(transpose(X_scaled) * loadings)
end

function tool_version()
    try
        v = Pkg.dependencies()
        for (_, info) in v
            if info.name == "OnlinePCA"
                return string(info.version)
            end
        end
    catch
    end
    return "unknown"
end

function write_output(path::AbstractString,
                      embedding, loadings, variance, variance_ratio,
                      cell_ids, gene_ids, args)
    h5open(path, "w") do h5
        write(h5, "embedding", Matrix{Float64}(embedding))
        write(h5, "loadings", Matrix{Float64}(loadings))
        write(h5, "variance", Vector{Float64}(variance))
        write(h5, "variance_ratio", Vector{Float64}(variance_ratio))
        write(h5, "cell_ids", String.(cell_ids))
        write(h5, "gene_ids", String.(gene_ids))

        attrs(h5)["format_version"] = OUTPUT_FORMAT_VERSION
        attrs(h5)["tool"] = TOOL
        attrs(h5)["tool_version"] = tool_version()
        attrs(h5)["solver"] = args["solver"]
        attrs(h5)["n_components"] = args["n_components"]
        attrs(h5)["random_seed"] = args["random_seed"]
    end
end

function main()
    s = build_pca_parser()
    args = parse_args(ARGS, s)

    println("Full command: ", join(ARGS, " "))
    for k in ("output_dir", "name", "normalized_h5", "selected_genes",
              "solver", "n_components", "random_seed")
        println("  $k: $(args[k])")
    end

    mkpath(args["output_dir"])

    selected = load_selected_genes(args["selected_genes"])
    println("  selected genes: $(length(selected))")

    X, gene_ids, cell_ids = load_subset_matrix(args["normalized_h5"], selected)
    println("  matrix (genes x cells): $(size(X))")

    X_scaled = scale_per_gene(X)

    workdir = mktempdir()
    binfile = joinpath(workdir, "normalized.zst")
    write_onlinepca_binary(X_scaled, binfile)
    OnlinePCA.sumr(binfile = binfile, outdir = workdir)

    loadings, variance, totalvar = run_onlinepca(
        binfile, workdir,
        args["solver"], args["n_components"], args["random_seed"],
    )
    embedding = compute_embedding(X_scaled, loadings)
    variance_ratio = totalvar > 0 ? variance ./ totalvar : zeros(length(variance))

    println("  embedding: $(size(embedding)), loadings: $(size(loadings))")

    out = joinpath(args["output_dir"],
                   "$(args["name"])_$(args["solver"])_n_$(args["n_components"]).h5")
    write_output(out, embedding, loadings, variance, variance_ratio,
                 cell_ids, gene_ids, args)
    println("  wrote: $out")

    rm(workdir; recursive = true, force = true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
