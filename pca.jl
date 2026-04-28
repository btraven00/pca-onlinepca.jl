#!/usr/bin/env julia
"""
PCA module (OnlinePCA.jl-backed) for omnibenchmark — raw-counts variant.

Output format: shared neutral HDF5 v1 (see `docs/pca_output.md`).
File: {output_dir}/{name}_{solver}_n_{n_components}.h5

Pipeline
--------
1. Read `/layers/counts` from the upstream h5ad (dense integer counts,
   cells x genes), plus `/obs/_index` (cell ids) and `/var/_index`
   (gene ids).
2. Apply `filtered.cellids` to drop unfiltered cells.
3. Apply `selected.genes` to drop unselected genes.
4. Build a sparse (genes x cells) Int matrix and write it to a temp
   TENx-format HDF5 that OnlinePCA.jl's `tenxpca` / `tenxsumr` can read.
5. `tenxsumr` computes per-row means; `tenxpca` runs the randomized SVD
   with the chosen `scale ∈ {sqrt, log, raw}`.
6. Write outputs to the neutral HDF5 spec.

OnlinePCA.jl's solvers are integer-only by design (memory-efficient
streaming; `loadchromium` reads matrix/data into a Vector{Int64}). That
is why this module operates on the raw count matrix from the datasets
stage instead of the normalized matrix that scanpy / scrapper consume.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using ArgParse
using HDF5
using SparseArrays
using Random
using Pkg
using GZip
using DelimitedFiles
using OnlinePCA

include(joinpath(@__DIR__, "src", "cli.jl"))
using .CLI

const OUTPUT_FORMAT_VERSION = "1"
const TOOL = "OnlinePCA.jl"
const TENX_GROUP = "matrix"

function load_lines(path::AbstractString)
    opener = endswith(path, ".gz") ? GZip.open : open
    return opener(path, "r") do io
        [strip(l) for l in eachline(io) if !isempty(strip(l))]
    end
end

"""
Read raw counts + ids from an h5ad written by anndataR.

Returns (X::Matrix{Int}, cell_ids::Vector{String}, gene_ids::Vector{String})
where X is (n_genes, n_cells) — h5ad on-disk is row-major (cells, genes),
which HDF5.jl reads column-major as (genes, cells).
"""
function load_h5ad_counts(path::AbstractString)
    h5open(path, "r") do h5
        haskey(h5, "layers") && haskey(h5["layers"], "counts") ||
            error("expected /layers/counts in $path")
        counts = read(h5["layers"]["counts"])
        cell_ids = String.(read(h5["obs"]["_index"]))
        gene_ids = String.(read(h5["var"]["_index"]))
        return counts, cell_ids, gene_ids
    end
end

"""
Subset (genes x cells) dense counts to selected genes/cells and return
a sparse genes-as-rows Int matrix plus aligned id vectors.
"""
function subset_counts(counts::AbstractMatrix, cell_ids::Vector{String},
                       gene_ids::Vector{String},
                       keep_cells::Vector{<:AbstractString},
                       keep_genes::Vector{<:AbstractString})
    n_genes_in, n_cells_in = size(counts)
    @assert length(gene_ids) == n_genes_in "gene_ids ($(length(gene_ids))) != counts rows ($n_genes_in)"
    @assert length(cell_ids) == n_cells_in "cell_ids ($(length(cell_ids))) != counts cols ($n_cells_in)"

    cell_set = Set(keep_cells)
    gene_set = Set(keep_genes)

    miss_c = setdiff(cell_set, Set(cell_ids))
    if !isempty(miss_c)
        sample = sort(collect(miss_c))[1:min(10, end)]
        error("$(length(miss_c)) filtered cell(s) not in h5ad; first $(length(sample)): $(sample)")
    end
    miss_g = setdiff(gene_set, Set(gene_ids))
    if !isempty(miss_g)
        sample = sort(collect(miss_g))[1:min(10, end)]
        error("$(length(miss_g)) selected gene(s) not in h5ad; first $(length(sample)): $(sample)")
    end

    gene_mask = [g in gene_set for g in gene_ids]
    cell_mask = [c in cell_set for c in cell_ids]

    sub = counts[gene_mask, cell_mask]                # (n_genes', n_cells')
    X = SparseMatrixCSC{Int64,Int64}(sparse(sub))
    kept_cells = cell_ids[cell_mask]
    kept_genes = gene_ids[gene_mask]

    # Drop cells that have zero total counts in the selected gene set.
    # OnlinePCA.jl's tenxnormalizex divides by per-cell sums for scale ∈
    # {sqrt, log}, so zero-sum cells produce NaN/Inf and crash LU.
    col_sums = vec(sum(X; dims = 1))
    nonempty = col_sums .> 0
    if !all(nonempty)
        n_dropped = count(.!nonempty)
        println("  dropped $n_dropped cell(s) with zero counts across selected genes")
        X = X[:, nonempty]
        kept_cells = kept_cells[nonempty]
    end

    return X, kept_cells, kept_genes
end

"""
Write a sparse (genes x cells) Int matrix to a TENx-format HDF5
that OnlinePCA.jl reads via `loadchromium`.
"""
function write_tenx_h5(path::AbstractString, X::SparseMatrixCSC,
                       gene_ids::Vector{String}, cell_ids::Vector{String})
    n_genes, n_cells = size(X)
    h5open(path, "w") do h5
        g = create_group(h5, TENX_GROUP)
        write(g, "data",    Vector{Int32}(X.nzval))
        write(g, "indices", Vector{UInt32}(X.rowval .- 1))
        write(g, "indptr",  Vector{UInt32}(X.colptr .- 1))
        write(g, "shape",   UInt32[n_genes, n_cells])
        write(g, "barcodes", cell_ids)
        feats = create_group(g, "features")
        write(feats, "id", gene_ids)
        write(feats, "name", gene_ids)
    end
end

function parse_solver(solver::AbstractString)
    @assert startswith(solver, "tenxpca_")
    scale = String(split(solver, "_"; limit=2)[2])
    @assert scale in ("sqrt", "log", "raw")
    rowmean = scale == "raw"  ? "Feature_Means.csv" :
              scale == "log"  ? "Feature_LogMeans.csv" :
                                "Feature_SqrtMeans.csv"
    return scale, rowmean
end

function tool_version()
    try
        for (_, info) in Pkg.dependencies()
            info.name == "OnlinePCA" && return string(info.version)
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
        write(h5, "loadings",  Matrix{Float64}(loadings))
        write(h5, "variance",  Vector{Float64}(variance))
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
    args = parse_args(ARGS, build_pca_parser())

    println("Full command: ", join(ARGS, " "))
    for k in ("output_dir", "name", "rawdata_h5ad", "filtered_cellids",
              "selected_genes", "solver", "n_components", "random_seed")
        println("  $k: $(args[k])")
    end

    Random.seed!(args["random_seed"])
    mkpath(args["output_dir"])

    keep_cells = load_lines(args["filtered_cellids"])
    keep_genes = load_lines(args["selected_genes"])
    println("  filtered cells: $(length(keep_cells))")
    println("  selected genes: $(length(keep_genes))")

    counts, all_cells, all_genes = load_h5ad_counts(args["rawdata_h5ad"])
    println("  raw counts (cells x genes): $(size(counts))")

    X, cell_ids, gene_ids = subset_counts(counts, all_cells, all_genes,
                                          keep_cells, keep_genes)
    n_genes, n_cells = size(X)
    println("  subset (genes x cells): ($n_genes, $n_cells), nnz=$(nnz(X))")

    workdir = mktempdir()
    try
        tenxfile = joinpath(workdir, "subset.h5")
        write_tenx_h5(tenxfile, X, gene_ids, cell_ids)

        scale, rowmean = parse_solver(args["solver"])

        OnlinePCA.tenxsumr(tenxfile = tenxfile, outdir = workdir,
                           group = TENX_GROUP)

        # NOTE: bypass the kwargs wrapper OnlinePCA.tenxpca(; ...) — at the
        # pinned commit (bfb2f75) it has a bug at tenxpca.jl:43 where it
        # references an undefined `pca` symbol (every sibling solver
        # initializes `pca = HALKO()` / `CCIPCA()` / etc.; tenxpca.jl is
        # missing the analogous `pca = TENXPCA()` line). Call the
        # positional-args inner method ourselves with a TENXPCA() instance.
        rml = joinpath(workdir, rowmean)
        rvl = ""        # rowvarlist: skip per-row variance scaling
        # colsumlist: per-cell totals (required for scale ∈ {sqrt, log};
        # tenxinit defaults colsumvec to zeros when this is "", which makes
        # tenxnormalizex divide by 0 → NaN/Inf → LU crash).
        csl = scale == "raw" ? "" : joinpath(workdir, "Sample_NoCounts.csv")
        dim = args["n_components"]
        noversamples = 5
        niter = 3
        chunksize = 5000
        cper = 1.0f0
        perm = false

        W, D, rowmeanvec, rowvarvec, colsumvec, N, M, TotalVar, idp =
            OnlinePCA.tenxinit(tenxfile, dim, chunksize, TENX_GROUP,
                               rml, rvl, csl, nothing, nothing, nothing,
                               cper, scale, perm)
        pca_alg = OnlinePCA.TENXPCA()
        out = OnlinePCA.tenxpca(tenxfile, nothing, scale, rml, rvl, csl,
                                dim, noversamples, niter, chunksize, nothing,
                                pca_alg, W, D, rowmeanvec, rowvarvec, colsumvec,
                                N, M, TotalVar, perm, idp, TENX_GROUP, cper)
        # tenxpca returns: (V, λ, U, Scores, ExpVar, TotalVar) where
        #   λ        : per-PC eigenvalue (variance per PC)        — out[2]
        #   U        : loadings, n_genes × dim                    — out[3]
        #   Scores   : PC scores, n_cells × dim                   — out[4]
        #   ExpVar   : SCALAR, sum(λ)/TotalVar (total fraction)   — out[5]
        #   TotalVar : SCALAR, total variance of scaled matrix    — out[6]
        Scores   = Matrix{Float64}(out[4])
        loadings = Matrix{Float64}(out[3])
        λ        = Vector{Float64}(out[2])
        TotalVar = Float64(out[6])

        variance       = λ
        variance_ratio = TotalVar > 0 ? λ ./ TotalVar : zeros(length(λ))

        println("  embedding: $(size(Scores)), loadings: $(size(loadings))")

        outpath = joinpath(args["output_dir"],
                           "$(args["name"])_$(args["solver"])_n_$(args["n_components"]).h5")
        write_output(outpath, Scores, loadings, variance, variance_ratio,
                     cell_ids, gene_ids, args)
        println("  wrote: $outpath")
    finally
        rm(workdir; recursive = true, force = true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
