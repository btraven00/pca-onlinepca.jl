#!/usr/bin/env julia
"""
PCA module (OnlinePCA.jl-backed) for omnibenchmark — raw-counts variant.

Outputs, per the PCA stage contract:
  {output_dir}/{name}_pcas.tsv       cell_id + one column per PC
  {output_dir}/{name}_loadings.tsv   gene_id + one column per PC
plus the shared neutral HDF5 v1 (see `docs/pca_output.md`), which is not
declared by the stage but is the only place the eigenvalues survive.

Pipeline
--------
1. Read `/layers/counts` from the upstream h5ad (dense integer counts,
   cells x genes), plus `/obs/_index` (cell ids) and `/var/_index`
   (gene ids).
2. Apply `filtered_cellids` to drop unfiltered cells.
3. Apply the HVG set — read from `matrix/genes` of
   `normalized_selected_h5` — to drop unselected genes.
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
Read the selected gene ids from a TENx-format H5 (`writeTENxMatrix(group="matrix")`,
genes x cells). Only `matrix/genes` is touched — the normalized values are
deliberately ignored, since OnlinePCA.jl scales raw counts itself.
"""
function load_selected_genes(path::AbstractString)
    h5open(path, "r") do h5
        haskey(h5, TENX_GROUP) && haskey(h5[TENX_GROUP], "genes") ||
            error("expected /$TENX_GROUP/genes in $path")
        return String.(read(h5[TENX_GROUP]["genes"]))
    end
end

"""
Read raw counts + ids from an h5ad.

Returns (X, cell_ids, gene_ids) with X (n_genes, n_cells).

`/layers/counts` is usually an AnnData CSR group (`encoding-type = csr_matrix`,
`shape = [n_obs, n_vars]`, i.e. cells x genes) rather than a dense array —
every fixture in this benchmark is stored that way, and HDF5.jl reads a group
as a Dict, so the dense path silently yields a Dict instead of a matrix.

The transpose is free: CSR of A is bit-for-bit CSC of Aᵀ. Reading
(data, indices, indptr) straight into a SparseMatrixCSC therefore gives the
(genes x cells) orientation we want, with no copy and no permutation.

Note `X` itself is absent from these files (`X: None` is legal AnnData); the
counts only ever live in the layer, so there is no fallback to fall back to.
"""
function load_h5ad_counts(path::AbstractString)
    h5open(path, "r") do h5
        haskey(h5, "layers") && haskey(h5["layers"], "counts") ||
            error("expected /layers/counts in $path")
        obj = h5["layers"]["counts"]
        cell_ids = String.(read(h5["obs"]["_index"]))
        gene_ids = String.(read(h5["var"]["_index"]))

        counts = if obj isa HDF5.Group
            enc = haskey(attrs(obj), "encoding-type") ? attrs(obj)["encoding-type"] : ""
            enc == "csr_matrix" ||
                error("/layers/counts is a $enc group; only csr_matrix (cells x genes) is supported")
            shape = attrs(obj)["shape"]                # [n_obs, n_vars] = [cells, genes]
            n_cells, n_genes = Int(shape[1]), Int(shape[2])
            data    = read(obj["data"])                # often float64 even for counts
            indices = read(obj["indices"])             # gene indices, 0-based
            indptr  = read(obj["indptr"])              # per-cell offsets, 0-based
            # CSR(cells x genes) == CSC(genes x cells): reuse the arrays as-is.
            SparseMatrixCSC{Int64,Int64}(n_genes, n_cells,
                                         Vector{Int64}(indptr) .+ 1,
                                         Vector{Int64}(indices) .+ 1,
                                         Vector{Int64}(data))
        else
            # Dense on-disk: h5ad is row-major (cells, genes), which HDF5.jl
            # reads column-major as (genes, cells) already.
            read(obj)
        end

        return counts, cell_ids, gene_ids
    end
end

"""
Read `/layers/counts` already subset to the kept cells and genes, without ever
holding the full matrix.

Why this exists: the raw h5ad carries every gene (36753 on the be1 fixture)
while FEAT keeps 2000, and the straightforward path -- read everything, then
mask -- peaked at ~540MB on that fixture before OnlinePCA had run at all. Two
costs stack there. The stored `data` is float64 and `indices` int32, so
converting both to Int64 allocates full-size copies, and `Vector{Int64}(x) .+ 1`
allocates a second one; and none of that work is needed for the ~5% of rows
that survive selection.

Here the nonzeros are read in cell-sized chunks and filtered on the way in, so
peak memory tracks the *kept* matrix rather than the stored one. Row indices
stay sorted within each column because `gene_row` is monotone in the original
gene order, which is what SparseMatrixCSC requires.

Returns (X::SparseMatrixCSC{Int64,Int64} genes x cells, kept_cells, kept_genes),
matching subset_counts.
"""
function load_counts_subset(path::AbstractString,
                            keep_cells::Vector{<:AbstractString},
                            keep_genes::Vector{<:AbstractString};
                            chunk_cells::Integer = 4096)
    h5open(path, "r") do h5
        haskey(h5, "layers") && haskey(h5["layers"], "counts") ||
            error("expected /layers/counts in $path")
        obj = h5["layers"]["counts"]
        cell_ids = String.(read(h5["obs"]["_index"]))
        gene_ids = String.(read(h5["var"]["_index"]))

        # Dense fallback: rare, and small enough that masking in memory is fine.
        if !(obj isa HDF5.Group)
            return subset_counts(read(obj), cell_ids, gene_ids, keep_cells, keep_genes)
        end
        enc = haskey(attrs(obj), "encoding-type") ? attrs(obj)["encoding-type"] : ""
        enc == "csr_matrix" ||
            error("/layers/counts is a $enc group; only csr_matrix (cells x genes) is supported")

        cell_set, gene_set = Set(keep_cells), Set(keep_genes)
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

        shape = attrs(obj)["shape"]
        n_cells, n_genes = Int(shape[1]), Int(shape[2])
        length(gene_ids) == n_genes || error("var/_index ($(length(gene_ids))) != shape genes ($n_genes)")
        length(cell_ids) == n_cells || error("obs/_index ($(length(cell_ids))) != shape cells ($n_cells)")

        # 0 marks a dropped gene; otherwise the row it becomes in the subset.
        gene_row = zeros(Int64, n_genes)
        r = 0
        for (j, g) in enumerate(gene_ids)
            if g in gene_set
                r += 1
                gene_row[j] = r
            end
        end
        kept_genes = gene_ids[gene_row .> 0]
        cell_keep  = [c in cell_set for c in cell_ids]
        n_keep     = count(cell_keep)

        indptr = Vector{Int64}(read(obj["indptr"]))    # 0-based, length n_cells+1
        d_set, i_set = obj["data"], obj["indices"]

        # One read per chunk. Survivors are counted from the copy already in
        # memory and the output grown by exactly that much, so there is neither
        # a second pass over the file nor the repeated doubling that a push!
        # loop would do -- guessing the final size up front is badly wrong
        # anyway, since selected genes are far denser than average ones.
        colptr = Vector{Int64}(undef, n_keep + 1); colptr[1] = 1
        rowval, nzval = Int64[], Int64[]

        col, at = 0, 0
        for lo in 1:chunk_cells:n_cells
            hi = min(lo + chunk_cells - 1, n_cells)
            a, b = indptr[lo] + 1, indptr[hi + 1]      # 1-based inclusive slice
            d, ix = if b >= a
                (d_set[a:b], i_set[a:b])               # hyperslab: this chunk only
            else
                (Float64[], Int32[])                   # all-empty cells in range
            end

            m = 0
            for c in lo:hi
                cell_keep[c] || continue
                @inbounds for p in (indptr[c] + 1):indptr[c + 1]
                    m += (gene_row[Int(ix[p - a + 1]) + 1] != 0)
                end
            end
            resize!(rowval, at + m); resize!(nzval, at + m)

            for c in lo:hi
                cell_keep[c] || continue
                col += 1
                @inbounds for p in (indptr[c] + 1):indptr[c + 1]
                    g = gene_row[Int(ix[p - a + 1]) + 1]
                    if g != 0
                        at += 1
                        rowval[at] = g
                        nzval[at]  = Int64(d[p - a + 1])
                    end
                end
                colptr[col + 1] = at + 1
            end
        end

        X = SparseMatrixCSC{Int64,Int64}(r, n_keep, colptr, rowval, nzval)
        kept_cells = cell_ids[cell_keep]

        # Same guard as subset_counts: OnlinePCA's tenxnormalizex divides by
        # per-cell sums for scale in {sqrt, log}, so a zero-sum cell gives
        # NaN/Inf and crashes LU.
        nonempty = vec(sum(X; dims = 1)) .> 0
        if !all(nonempty)
            println("  dropped $(count(.!nonempty)) cell(s) with zero counts across selected genes")
            X = X[:, nonempty]
            kept_cells = kept_cells[nonempty]
        end

        return X, kept_cells, kept_genes
    end
end

"""
Subset (genes x cells) dense counts to selected genes/cells and return
a sparse genes-as-rows Int matrix plus aligned id vectors.

Kept for the dense-on-disk path; the CSR path uses load_counts_subset.
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
Mean per-cell total over the selected genes — the target library size handed to
OnlinePCA as `cper`. Matches `centerSizeFactors()` in the NORM stage, which
centers size factors at 1 (equivalently: rescale every cell to the mean library
size) before the log.

Computed over the *subset* matrix, because `Sample_NoCounts.csv` from
`tenxsumr`/`sumr` is likewise per-cell sums of the selected genes only.
"""
mean_colsum(X::SparseMatrixCSC) = Float32(nnz(X) == 0 ? 1.0 : sum(X) / size(X, 2))

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

"""
Parse a `{algorithm}_{scale}` solver token. Returns (algorithm, scale, rowmean_csv).

The per-row-mean file name depends on the scale because OnlinePCA.sumr writes
one CSV per transform (Feature_Means / Feature_LogMeans / Feature_SqrtMeans
/ Feature_FTTMeans). Each algorithm only supports a subset of scales; this
function also validates that the combination is admissible.
"""
function parse_solver(solver::AbstractString)
    parts = split(solver, "_"; limit=2)
    @assert length(parts) == 2 "solver must be of form {algorithm}_{scale}, got: $solver"
    algorithm = String(parts[1])
    scale = String(parts[2])

    rowmean = scale == "raw"  ? "Feature_Means.csv" :
              scale == "log"  ? "Feature_LogMeans.csv" :
              scale == "sqrt" ? "Feature_SqrtMeans.csv" :
              scale == "ftt"  ? "Feature_FTTMeans.csv" :
                                error("unknown scale: $scale")

    valid_scales = Dict("tenxpca" => ("sqrt", "log", "raw"),
                        "algorithm971" => ("log", "ftt", "raw"))
    @assert haskey(valid_scales, algorithm) "unknown algorithm: $algorithm"
    @assert scale in valid_scales[algorithm] "algorithm $algorithm does not support scale $scale"

    return algorithm, scale, rowmean
end

"""
Stream `/layers/counts` straight from the h5ad into the TENx file `tenxpca`
reads, never holding the whole matrix.

This is what makes the module out-of-core. OnlinePCA's factorisation already
streams -- `tenxpca` adds 0MB -- but everything in front of it did not: the
matrix was materialised as a SparseMatrixCSC and only then written out, so
peak grew with the data no matter how well the solver behaved.

The transposition is free, which is what makes this bookkeeping rather than a
redesign. h5ad stores CSR (cells x genes): one contiguous run per cell, listing
gene indices in order. TENx wants CSC (genes x cells): one contiguous run per
cell, listing gene row-indices in order. Same bytes, same order -- each cell's
run is appended as one column. No transpose, no re-sort, and the gene remap is
monotone so row indices stay sorted within a column.

Two passes, both O(chunk) in memory:

  1. read `data`+`indices` per cell-chunk; count kept nonzeros per cell, sum
     each cell over the kept genes, and accumulate the per-gene transformed
     sums. That fixes the exact nnz and cell count, so pass 2 can write into
     fixed-size datasets and no extendible-dataset dance is needed.
  2. read again and write the `data`/`indices` slabs.

Cells whose kept-gene total is zero are dropped, matching what subset_counts
did after the fact: OnlinePCA's tenxnormalizex divides by the per-cell sum for
scale in {sqrt, log}, so a zero-sum cell yields NaN/Inf and crashes LU. Their
contribution to the gene sums is zero under every supported transform, so
accumulating before the drop is decided is safe -- only the divisor changes.

Returns a NamedTuple with everything downstream needs in place of the matrix:
(n_genes, n_cells, nnz, nocounts, gene_means, kept_cells, kept_genes).
"""
function stream_h5ad_to_tenx(path::AbstractString, tenxfile::AbstractString,
                             keep_cells::Vector{<:AbstractString},
                             keep_genes::Vector{<:AbstractString},
                             scale::AbstractString;
                             chunk_cells::Integer = 512)
    scale in ("raw", "log", "sqrt") ||
        error("unsupported scale for the streaming path: $scale")
    transform = scale == "raw"  ? identity :
                scale == "log"  ? (x -> log10(x + 1)) :
                                  sqrt

    h5open(path, "r") do h5
        haskey(h5, "layers") && haskey(h5["layers"], "counts") ||
            error("expected /layers/counts in $path")
        obj = h5["layers"]["counts"]
        obj isa HDF5.Group ||
            error("/layers/counts is dense; the streaming path needs csr_matrix")
        enc = haskey(attrs(obj), "encoding-type") ? attrs(obj)["encoding-type"] : ""
        enc == "csr_matrix" ||
            error("/layers/counts is a $enc group; only csr_matrix (cells x genes) is supported")

        cell_ids = String.(read(h5["obs"]["_index"]))
        gene_ids = String.(read(h5["var"]["_index"]))
        cell_set, gene_set = Set(keep_cells), Set(keep_genes)
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

        shape = attrs(obj)["shape"]
        n_cells_in, n_genes_in = Int(shape[1]), Int(shape[2])
        length(gene_ids) == n_genes_in || error("var/_index != shape genes")
        length(cell_ids) == n_cells_in || error("obs/_index != shape cells")

        gene_row = zeros(Int64, n_genes_in)          # 0 = dropped
        r = 0
        for (j, g) in enumerate(gene_ids)
            if g in gene_set
                r += 1
                gene_row[j] = r
            end
        end
        kept_genes = gene_ids[gene_row .> 0]
        cell_keep  = [c in cell_set for c in cell_ids]

        indptr = Vector{Int64}(read(obj["indptr"]))
        d_set, i_set = obj["data"], obj["indices"]
        chunks = [(lo, min(lo + chunk_cells - 1, n_cells_in))
                  for lo in 1:chunk_cells:n_cells_in]

        # ---- pass 1: sizes, per-cell sums, per-gene sums -------------------
        per_cell_nnz = zeros(Int64, n_cells_in)
        per_cell_sum = zeros(Int64, n_cells_in)
        gene_sums    = zeros(Float64, r)
        for (lo, hi) in chunks
            a, b = indptr[lo] + 1, indptr[hi + 1]
            b >= a || continue
            d, ix = d_set[a:b], i_set[a:b]
            for c in lo:hi
                cell_keep[c] || continue
                @inbounds for p in (indptr[c] + 1):indptr[c + 1]
                    g = gene_row[Int(ix[p - a + 1]) + 1]
                    g == 0 && continue
                    v = Int64(d[p - a + 1])
                    per_cell_nnz[c] += 1
                    per_cell_sum[c] += v
                    gene_sums[g]    += transform(v)
                end
            end
        end

        keep = [cell_keep[c] && per_cell_sum[c] > 0 for c in 1:n_cells_in]
        n_dropped = count(cell_keep) - count(keep)
        n_dropped > 0 &&
            println("  dropped $n_dropped cell(s) with zero counts across selected genes")
        kept_cells = cell_ids[keep]
        n_keep     = length(kept_cells)
        total      = sum(per_cell_nnz[keep])

        # TENx indptr is 0-based, one entry per kept cell plus the tail.
        colptr = Vector{UInt32}(undef, n_keep + 1); colptr[1] = 0
        k = 0
        for c in 1:n_cells_in
            keep[c] || continue
            k += 1
            colptr[k + 1] = colptr[k] + UInt32(per_cell_nnz[c])
        end

        # ---- pass 2: write ------------------------------------------------
        h5open(tenxfile, "w") do out
            g = create_group(out, TENX_GROUP)
            d_out = create_dataset(g, "data",    Int32,  (total,))
            i_out = create_dataset(g, "indices", UInt32, (total,))
            at = 0
            for (lo, hi) in chunks
                a, b = indptr[lo] + 1, indptr[hi + 1]
                b >= a || continue
                d, ix = d_set[a:b], i_set[a:b]
                buf_d, buf_i = Int32[], UInt32[]
                for c in lo:hi
                    keep[c] || continue
                    @inbounds for p in (indptr[c] + 1):indptr[c + 1]
                        gg = gene_row[Int(ix[p - a + 1]) + 1]
                        gg == 0 && continue
                        push!(buf_d, Int32(d[p - a + 1]))
                        push!(buf_i, UInt32(gg - 1))      # TENx rows are 0-based
                    end
                end
                if !isempty(buf_d)
                    d_out[(at + 1):(at + length(buf_d))] = buf_d
                    i_out[(at + 1):(at + length(buf_i))] = buf_i
                    at += length(buf_d)
                end
            end
            at == total || error("wrote $at nonzeros, expected $total")
            write(g, "indptr",   colptr)
            write(g, "shape",    UInt32[r, n_keep])
            write(g, "barcodes", kept_cells)
            feats = create_group(g, "features")
            write(feats, "id",   kept_genes)
            write(feats, "name", kept_genes)
        end

        return (n_genes = r, n_cells = n_keep, nnz = total,
                nocounts = UInt32.(per_cell_sum[keep]),
                gene_means = gene_sums ./ n_keep,
                kept_cells = kept_cells, kept_genes = kept_genes)
    end
end

"""
Write a sparse (genes x cells) integer matrix in MatrixMarket coordinate
format. The header is the bare minimum that OnlinePCA.mm2bin accepts: one
banner line followed by `nrows ncols nnz` and the triplets. No comment
lines (mm2bin parses dimensions from line 2 unconditionally).
"""
function write_mm(path::AbstractString, X::SparseMatrixCSC)
    n_rows, n_cols = size(X)
    nz = nnz(X)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate integer general")
        println(io, "$n_rows $n_cols $nz")
        rows = rowvals(X)
        vals = nonzeros(X)
        for col in 1:n_cols
            for k in nzrange(X, col)
                println(io, "$(rows[k]) $col $(vals[k])")
            end
        end
    end
end

"""
Gene-chunk size for tenxinit/tenxpca.

`chunksize` is what makes OnlinePCA's out-of-core path actually stream: each
chunk is a gene range, and `loadchromium` holds that range's nonzeros across
all cells. With the previous hardcoded 5000 against a 2000-gene subset there
was a single chunk, i.e. the whole matrix in memory and no streaming at all.
Measured on sc-mix, tenxpca's own peak:

    chunksize 5000  +491MB / 8.4s     501  +107MB / 12.1s
              1001  +277MB / 9.6s     251   +35MB / 16.9s

Eight chunks trades roughly 2x the tenxpca time for ~14x less memory, which is
the right way round for a step that was 2GB.

The `+ 1` is not cosmetic. Upstream computes `lasti = fld(N, chunksize) + 1`,
so when chunksize divides N exactly it emits one chunk too many, starting past
the last gene; `loadchromium` then hits `@assert minimum(newindices) >= startp`
on an empty vector and dies with "reducing over an empty collection"
(Utils.jl:143). Landing on a size that does not divide N avoids the empty tail
chunk entirely.
"""
function gene_chunksize(n_genes::Integer; target_chunks::Integer = 8)
    cs = max(64, cld(n_genes, target_chunks))
    # cs == n_genes is the worst case, not a safe one: fld(N, N) + 1 == 2, and
    # the second chunk is empty. Going strictly above N gives fld == 0, hence a
    # single chunk, which is what upstream's own default relied on.
    cs >= n_genes && return n_genes + 1
    return n_genes % cs == 0 ? cs + 1 : cs
end

"""
Write the two summary files `tenxpca` actually reads, in place of calling
`OnlinePCA.tenxsumr`.

This is a workaround for upstream, not for anything in this module. Measured on
sc-mix (2000 genes x 3918 cells, a 22MB TENx file), `tenxsumr` adds **+998MB**
of peak RSS while `tenxpca` itself adds none. Two upstream causes, both in
OnlinePCA/src/tenxsumr.jl:

  * `tenxstats` computes 25 statistics -- means and variances, each raw/log/
    sqrt, each again under CPM/CPT/CPMED, plus CV2 -- and holds twelve full
    materialised copies of the chunk at once (X, logX, sqrtX, cpmX, logcpmX,
    sqrtcpmX, cptX, ..., sqrtcpmedX). With the default chunksize of 5000 and a
    2000-gene subset, the "chunk" is the entire matrix.
  * the `1e6 .* X ./ nc'` scalings promote those copies to Float64.

Of the 25, `run_tenxpca` reads exactly two, and X is already in memory here.

Definitions are copied from `tenxstats`/`tenxnocounts` so the files stay
byte-compatible with what tenxsumr would have written:
  Sample_NoCounts.csv   UInt32 per-cell totals (sum over rows)
  Feature_*Means.csv    row means over ALL cells, zeros included, under the
                        scale's transform: identity, log10(x+1), or sqrt(x)
"""
function write_sumr_csvs(X::SparseMatrixCSC, workdir::AbstractString,
                         scale::AbstractString, rowmean::AbstractString)
    n_genes, n_cells = size(X)

    # tenxnocounts accumulates into a UInt32 vector; match the type so the CSV
    # is written as integers rather than 1234.0.
    nc = UInt32.(vec(sum(X, dims = 1)))
    writedlm(joinpath(workdir, "Sample_NoCounts.csv"), nc, ',')

    nz = if scale == "raw"
        X.nzval
    elseif scale == "log"
        log10.(X.nzval .+ 1)          # sparseLog10
    elseif scale == "sqrt"
        sqrt.(X.nzval)
    else
        error("unsupported scale for the direct summary path: $scale")
    end
    T = SparseMatrixCSC(n_genes, n_cells, X.colptr, X.rowval, nz)
    # mean over dims=2 counts the structural zeros, hence /n_cells not /nnz
    writedlm(joinpath(workdir, rowmean), vec(sum(T, dims = 2)) ./ n_cells, ',')
end

"""
Out-of-core variant of run_tenxpca: streams the h5ad into the TENx file and
runs the factorisation from there, so the count matrix is never in memory.

Same result as the in-memory path (the TENx file and both summary CSVs are
byte-identical, verified for raw/log/sqrt); the difference is only where the
data lives. Returns the usual four values plus the ids and dimensions that
main would otherwise have taken from X.
"""
function run_tenxpca_streamed(h5ad::AbstractString,
                              keep_cells::Vector{<:AbstractString},
                              keep_genes::Vector{<:AbstractString},
                              scale::AbstractString, rowmean::AbstractString,
                              dim::Integer, workdir::AbstractString;
                              gene_chunks::Integer = 8)
    tenxfile = joinpath(workdir, "subset.h5")
    st = stream_h5ad_to_tenx(h5ad, tenxfile, keep_cells, keep_genes, scale)

    writedlm(joinpath(workdir, "Sample_NoCounts.csv"), st.nocounts, ',')
    writedlm(joinpath(workdir, rowmean), st.gene_means, ',')

    rml = joinpath(workdir, rowmean)
    rvl = ""
    csl = scale == "raw" ? "" : joinpath(workdir, "Sample_NoCounts.csv")
    # mean_colsum(X) == mean of the per-cell totals, which pass 1 already has.
    cper = Float32(st.nnz == 0 ? 1.0 : sum(st.nocounts) / st.n_cells)
    chunksize = gene_chunksize(st.n_genes; target_chunks = gene_chunks)

    W, D, rowmeanvec, rowvarvec, colsumvec, N, M, TotalVar, idp =
        OnlinePCA.tenxinit(tenxfile, dim, chunksize, TENX_GROUP,
                           rml, rvl, csl, nothing, nothing, nothing,
                           cper, scale, false)
    out = OnlinePCA.tenxpca(tenxfile, nothing, scale, rml, rvl, csl,
                            dim, 5, 3, chunksize, nothing,
                            OnlinePCA.TENXPCA(), W, D, rowmeanvec, rowvarvec,
                            colsumvec, N, M, TotalVar, false, idp, TENX_GROUP, cper)
    return Matrix{Float64}(out[1]), Matrix{Float64}(out[3]),
           Vector{Float64}(out[2]), Float64(out[6]), st
end

"""
Run OnlinePCA.tenxpca on a sparse (genes x cells) Int matrix.

Bypasses the broken kwargs wrapper at OnlinePCA commit bfb2f75 (see BUG.md).
Returns (V, loadings, variance_per_pc, total_variance) — see `scanpy_scores`
for why V (unit-norm) rather than OnlinePCA's own `Scores`.

Kept for algorithm971 and for the equivalence tests; the tenxpca path now goes
through run_tenxpca_streamed.
"""
function run_tenxpca(X::SparseMatrixCSC, gene_ids::Vector{String},
                     cell_ids::Vector{String}, scale::AbstractString,
                     rowmean::AbstractString, dim::Integer,
                     workdir::AbstractString)
    tenxfile = joinpath(workdir, "subset.h5")
    write_tenx_h5(tenxfile, X, gene_ids, cell_ids)

    # was: OnlinePCA.tenxsumr(tenxfile = tenxfile, outdir = workdir,
    #                         group = TENX_GROUP)
    # See write_sumr_csvs: same two files, without tenxsumr's 23 unused
    # statistics and its twelve simultaneous copies of the matrix.
    write_sumr_csvs(X, workdir, scale, rowmean)

    rml = joinpath(workdir, rowmean)
    rvl = ""        # rowvarlist: skip per-row variance scaling
    # colsumlist: per-cell totals (required for scale ∈ {sqrt, log};
    # tenxinit defaults colsumvec to zeros when this is "", which makes
    # tenxnormalizex divide by 0 → NaN/Inf → LU crash).
    csl = scale == "raw" ? "" : joinpath(workdir, "Sample_NoCounts.csv")
    noversamples = 5
    niter = 3
    chunksize = gene_chunksize(size(X, 1))
    # cper is the target library size: OnlinePCA computes
    #   cper * x / colsum   then   log10(. + pseudocount) / sqrt(.)
    # (Utils.jl:1412-1422). With cper = 1 the values land around 1e-4, so
    # log10(1e-4 + 1) ~ 4e-5 and the VST is very nearly the identity -- the
    # "log" and "sqrt" scales collapse toward raw, on a hugely shrunken scale.
    # The NORM stage uses centerSizeFactors(), i.e. size factors centered at 1,
    # which is exactly rescaling to the MEAN library size before the log. Match
    # that so this branch differs from the others in solver, not in target scale.
    cper = mean_colsum(X)
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
    # tenxpca returns: (V, λ, U, Scores, ExpVar, TotalVar)
    return Matrix{Float64}(out[1]),  # V, cells x dim, unit-norm columns
           Matrix{Float64}(out[3]),  # loadings (U)
           Vector{Float64}(out[2]),  # variance per PC (λ)
           Float64(out[6])           # total variance
end

"""
Run OnlinePCA.algorithm971 (Linderman/Li/Rokhlin 2017 randomized SVD
designed for large sparse single-cell matrices) on a sparse (genes x cells)
Int matrix.

I/O path:
  X  →  MatrixMarket file  →  mm2bin (.zst)  →  sumr (per-row/col stats)
                                                     ↓
                                              algorithm971

Returns (V, loadings, variance_per_pc, total_variance).
"""
function run_algorithm971(X::SparseMatrixCSC, scale::AbstractString,
                          rowmean::AbstractString, dim::Integer,
                          workdir::AbstractString)
    mmfile  = joinpath(workdir, "subset.mtx")
    binfile = joinpath(workdir, "subset.zst")
    write_mm(mmfile, X)
    OnlinePCA.mm2bin(mmfile = mmfile, binfile = binfile)

    # sumr in sparse_mm mode reads the (row, col, val) triplets emitted by
    # mm2bin and writes per-row/per-column statistics CSVs to outdir.
    OnlinePCA.sumr(binfile = binfile, outdir = workdir, mode = "sparse_mm")

    rml = joinpath(workdir, rowmean)
    rvl = ""    # rowvarlist: skip per-row variance scaling
    csl = scale == "raw" ? "" : joinpath(workdir, "Sample_NoCounts.csv")
    noversamples = 5
    niter = 3
    cper = mean_colsum(X)   # see run_tenxpca for why not 1.0

    out = OnlinePCA.algorithm971(
        input = binfile,
        outdir = nothing,
        scale = scale,
        pseudocount = 1.0f0,
        rowmeanlist = rml,
        rowvarlist = rvl,
        colsumlist = csl,
        dim = dim,
        noversamples = noversamples,
        niter = niter,
        perm = false,
        cper = cper,
    )
    # algorithm971 returns: (V, λ, U, Scores, ExpVar, TotalVar) — same layout
    # as tenxpca per the docstring.
    return Matrix{Float64}(out[1]),  # V, cells x dim, unit-norm columns
           Matrix{Float64}(out[3]),  # loadings (U)
           Vector{Float64}(out[2]),  # variance per PC (λ)
           Float64(out[6])           # total variance
end

"""
Convert OnlinePCA's right singular vectors to the embedding convention used by
scanpy and scrapper.

OnlinePCA (`src/algorithm971.jl:180-190`, and identically in tenxpca) computes

    W, σ, V = svd(B);  λ = σ .* σ ./ M;  Scores[:, n] = λ[n] .* V[:, n]

with M = number of cells. scanpy/scrapper instead write σ·V. Emitting λ·V would
reweight the PCs by an extra factor of σ/M each, which propagates straight into
the kNN graph and the clustering metrics — the benchmark would then be partly
measuring the scaling convention rather than the solver. So recover σ and apply
it: σ = sqrt(λ · M).

(OnlinePCA divides by M where a sample covariance would use M-1; the resulting
sqrt(M/(M-1)) discrepancy is a uniform scalar across all PCs, so it shifts no
neighbour ordering.)
"""
function scanpy_scores(V::AbstractMatrix, λ::AbstractVector, n_cells::Integer)
    σ = sqrt.(λ .* n_cells)
    return V .* σ'
end

"""
Write a matrix as the benchmark's embedding/loadings TSV: one header line
(`row_label`, then `dim_1..dim_k`) followed by one row per entry prefixed with
its id. Matches scanpy's `src/writers.py::_write_tsv` and scrapper's
`fwrite(data.frame(cell_id = ..., ...))`, which is what the downstream readers
expect (`knn.py` skips one header row and takes column 0 as the id;
`metrics/embedding` joins on the `cell_id` column by name).
"""
function write_tsv(path::AbstractString, M::AbstractMatrix,
                   row_ids::Vector{String}, row_label::AbstractString)
    n_rows, n_cols = size(M)
    @assert length(row_ids) == n_rows "row_ids ($(length(row_ids))) != rows ($n_rows)"
    open(path, "w") do io
        println(io, row_label, "\t", join(("dim_$i" for i in 1:n_cols), "\t"))
        for i in 1:n_rows
            println(io, row_ids[i], "\t", join(view(M, i, :), "\t"))
        end
    end
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
        # Recorded because it is not numerically inert: chunk boundaries change
        # the Float32 accumulation order, so two runs differing only in
        # gene_chunks are not bit-comparable.
        attrs(h5)["gene_chunks"] = args["gene_chunks"]
    end
end

function main()
    args = parse_args(ARGS, build_pca_parser())

    println("Full command: ", join(ARGS, " "))
    for k in ("output_dir", "name", "rawdata_h5ad", "filtered_cellids",
              "normalized_selected_h5", "solver", "n_components", "random_seed")
        println("  $k: $(args[k])")
    end

    Random.seed!(args["random_seed"])
    mkpath(args["output_dir"])

    keep_cells = load_lines(args["filtered_cellids"])
    keep_genes = load_selected_genes(args["normalized_selected_h5"])
    println("  filtered cells: $(length(keep_cells))")
    println("  selected genes: $(length(keep_genes))")

    algorithm, scale, rowmean = parse_solver(args["solver"])
    dim = args["n_components"]
    println("  dispatch: algorithm=$algorithm, scale=$scale")

    workdir = mktempdir()
    try
        # tenxpca streams: the h5ad goes straight to the TENx file it reads, so
        # the count matrix is never materialised. algorithm971 still needs X in
        # memory -- it is blocked on an unrelated format mismatch (see BUG.md),
        # so there is nothing to gain by converting it yet.
        local cell_ids, gene_ids, n_genes, n_cells
        V, loadings, λ, TotalVar = if algorithm == "tenxpca"
            v, l, lam, tv, st = run_tenxpca_streamed(args["rawdata_h5ad"],
                                                     keep_cells, keep_genes,
                                                     scale, rowmean, dim, workdir;
                                                     gene_chunks = args["gene_chunks"])
            cell_ids, gene_ids = st.kept_cells, st.kept_genes
            n_genes, n_cells = st.n_genes, st.n_cells
            println("  subset (genes x cells): ($n_genes, $n_cells), nnz=$(st.nnz)")
            (v, l, lam, tv)
        elseif algorithm == "algorithm971"
            X, cells, genes = load_counts_subset(args["rawdata_h5ad"],
                                                 keep_cells, keep_genes)
            cell_ids, gene_ids = cells, genes
            n_genes, n_cells = size(X)
            println("  subset (genes x cells): ($n_genes, $n_cells), nnz=$(nnz(X))")
            run_algorithm971(X, scale, rowmean, dim, workdir)
        else
            error("unhandled algorithm: $algorithm")
        end

        embedding      = scanpy_scores(V, λ, n_cells)
        variance       = λ
        variance_ratio = TotalVar > 0 ? λ ./ TotalVar : zeros(length(λ))

        println("  embedding: $(size(embedding)), loadings: $(size(loadings))")

        # The two stage-declared outputs. Cell ids are the post-drop vector, so
        # a zero-count cell simply doesn't appear; every downstream consumer
        # joins on cell_id (metrics/{embedding,graph,cluster}) or reads the ids
        # off the embedding itself (scanpy knn.py).
        pcas_path = joinpath(args["output_dir"], "$(args["name"])_pcas.tsv")
        loadings_path = joinpath(args["output_dir"], "$(args["name"])_loadings.tsv")
        write_tsv(pcas_path, embedding, cell_ids, "cell_id")
        write_tsv(loadings_path, loadings, gene_ids, "gene_id")
        println("  wrote: $pcas_path")
        println("  wrote: $loadings_path")

        # Undeclared, but the only place the eigenvalues and total variance
        # survive; the TSV contract carries neither.
        outpath = joinpath(args["output_dir"],
                           "$(args["name"])_$(args["solver"])_n_$(args["n_components"]).h5")
        write_output(outpath, embedding, loadings, variance, variance_ratio,
                     cell_ids, gene_ids, args)
        println("  wrote: $outpath")
    finally
        rm(workdir; recursive = true, force = true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
