#!/usr/bin/env julia
# Self-check for the pure parts of pca.jl: the score rescale, the TSV writer,
# and the HVG-set reader. Run with:  julia --project=. test_pca.jl
#
# Deliberately not a test framework and not per-function coverage — these are
# the three pieces where a silent wrong answer would propagate into the
# benchmark's metrics instead of crashing.

using LinearAlgebra
using HDF5
using SparseArrays

include(joinpath(@__DIR__, "pca.jl"))

# --- scanpy_scores: recover σ from λ -----------------------------------------
# OnlinePCA reports λ = σ²/M and Scores = λ·V; scanpy/scrapper write σ·V.
# The failure mode this catches is inverting it wrong (sqrt(λ/M), λ*M, ...),
# which yields a plausible matrix that silently reweights every PC.
let
    M = 40                                  # cells
    B = randn(6, M)
    W, σ, V = svd(B)
    λ = σ .* σ ./ M

    got = scanpy_scores(V, λ, M)
    @assert size(got) == size(V) "shape changed: $(size(got)) vs $(size(V))"
    @assert isapprox(got, V .* σ'; rtol = 1e-10) "scanpy_scores != V .* σ'"

    # and it is NOT OnlinePCA's own Scores, unless σ happens to equal λ
    native = V .* λ'
    @assert !isapprox(got, native; rtol = 1e-6) "rescale was a no-op"
end

# --- scanpy_scores against a reference PCA -----------------------------------
# Independent path: centre a matrix, take its SVD directly, and confirm the
# embedding matches what scanpy would write (up to per-PC sign).
let
    n_cells, n_genes, k = 50, 12, 4
    A = randn(n_cells, n_genes)
    Ac = A .- sum(A; dims = 1) ./ n_cells        # centre columns
    U, σ_ref, _ = svd(Ac)
    ref = U[:, 1:k] .* σ_ref[1:k]'               # scanpy convention

    λ = (σ_ref[1:k] .^ 2) ./ n_cells             # as OnlinePCA reports it
    got = scanpy_scores(U[:, 1:k], λ, n_cells)

    for j in 1:k
        c = dot(got[:, j], ref[:, j]) / (norm(got[:, j]) * norm(ref[:, j]))
        @assert isapprox(abs(c), 1.0; atol = 1e-8) "PC $j not collinear (|cos|=$c)"
        @assert isapprox(norm(got[:, j]), norm(ref[:, j]); rtol = 1e-8) "PC $j wrong scale"
    end
end

# --- write_tsv: the on-disk contract downstream readers assume ---------------
# knn.py: skip 1 header row, column 0 is the id, rest is the embedding.
# metrics/embedding: read with header, join on the "cell_id" column by name.
let
    dir = mktempdir()
    try
        emb = [1.5 -2.0; 3.25 4.0; 0.0 1.0]
        ids = ["cell_a", "cell_b", "cell_c"]
        path = joinpath(dir, "x_pcas.tsv")
        write_tsv(path, emb, ids, "cell_id")

        lines = readlines(path)
        @assert length(lines) == 4 "expected header + 3 rows, got $(length(lines))"
        @assert lines[1] == "cell_id\tdim_1\tdim_2" "bad header: $(lines[1])"

        # header field count must equal data field count, or polars mis-parses
        nhead = length(split(lines[1], '\t'))
        for l in lines[2:end]
            @assert length(split(l, '\t')) == nhead "ragged row: $l"
        end

        for (i, l) in enumerate(lines[2:end])
            f = split(l, '\t')
            @assert f[1] == ids[i] "row $i id: $(f[1])"
            @assert parse(Float64, f[2]) == emb[i, 1] "row $i col 1"
            @assert parse(Float64, f[3]) == emb[i, 2] "row $i col 2"
        end

        # loadings share the layout, only the label differs
        lp = joinpath(dir, "x_loadings.tsv")
        write_tsv(lp, emb, ids, "gene_id")
        @assert startswith(readlines(lp)[1], "gene_id\t")
    finally
        rm(dir; recursive = true, force = true)
    end
end

# --- load_selected_genes: HVG set comes from matrix/genes --------------------
let
    dir = mktempdir()
    try
        path = joinpath(dir, "sel.h5")
        genes = ["ENSG1", "ENSG2", "ENSG3"]
        h5open(path, "w") do h5
            g = create_group(h5, "matrix")
            write(g, "genes", genes)
            write(g, "barcodes", ["c1", "c2"])
            write(g, "data", Int32[1, 2])     # values must be ignored
        end
        @assert load_selected_genes(path) == genes

        bad = joinpath(dir, "bad.h5")
        h5open(bad, "w") do h5
            create_group(h5, "matrix")
        end
        threw = false
        try; load_selected_genes(bad); catch; threw = true; end
        @assert threw "missing matrix/genes should error, not return empty"
    finally
        rm(dir; recursive = true, force = true)
    end
end

println("all checks passed")
