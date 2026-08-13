"""
Argument parser for omnibenchmark OnlinePCA.jl module.

Conventions:
- All arguments are required. No defaults — callers (omnibenchmark configs)
  must pass everything explicitly so runs are fully reproducible.
- ArgParse rejects unknown flags by default; we rely on that for strictness.

OnlinePCA.jl is integer-counts-only (its loaders parse matrix/data
into a Vector{Int}). We therefore consume the raw count matrix from
the upstream `datasets` stage (h5ad `/layers/counts`), apply filtering
+ gene selection, and feed integer counts to the chosen algorithm.

Flag names are the benchmark's stage-input ids verbatim — omnibenchmark
emits `--{output_id}` (backend/snakemake.py) — so they must stay in sync
with the PCA stage's `inputs:` in the plan.

The HVG set comes from `matrix/genes` of `normalized_selected_h5` rather
than a dedicated gene-list output: no stage emits one, and borrowing the
row names keeps this module on the *same* gene set as the modules that
consume the normalized matrix, which is what makes the comparison fair.
Only the row names are read — never the normalized values.

Solver tokens are of the form `{algorithm}_{scale}`:
  - algorithm ∈ {tenxpca, algorithm971}
  - scale     ∈ {log, ftt, sqrt, raw}  (algorithm-specific subset; see below)

The `--solver` value selects both the algorithm and the on-the-fly
scale variant applied by OnlinePCA during the SVD pass.
"""

module CLI

using ArgParse

export build_pca_parser, VALID_SOLVERS

# tenxpca scales:        sqrt, log, raw   (per OnlinePCA.tenxpca docstring)
# algorithm971 scales:   log, ftt, raw    (per OnlinePCA.algorithm971 docstring)
const VALID_SOLVERS = [
    "tenxpca_sqrt", "tenxpca_log", "tenxpca_raw",
    "algorithm971_log", "algorithm971_ftt", "algorithm971_raw",
]

function build_pca_parser()
    s = ArgParseSettings(description = "OmniBenchmark PCA module (OnlinePCA.jl, raw-counts)",
                         autofix_names = false)

    @add_arg_table! s begin
        "--output_dir"
            help = "Output directory for results"
            arg_type = String
            required = true
        "--name"
            help = "Module name/identifier"
            arg_type = String
            required = true
        "--rawdata_h5ad"
            help = "h5ad with /layers/counts (cells x genes, integer)"
            arg_type = String
            required = true
        "--filtered_cellids"
            help = "Gzipped text file of kept cell barcodes (one per line)"
            arg_type = String
            required = true
        "--normalized_selected_h5"
            help = "TENx-format H5; only matrix/genes is read, for the HVG set"
            arg_type = String
            required = true
        "--solver"
            help = "OnlinePCA.jl scale variant: " * join(VALID_SOLVERS, ", ")
            arg_type = String
            required = true
            range_tester = (x -> x in VALID_SOLVERS)
        "--n_components"
            help = "Number of principal components to compute"
            arg_type = Int
            required = true
        "--random_seed"
            help = "Seed for randomized solvers (and for reproducibility)"
            arg_type = Int
            required = true
    end

    return s
end

end # module
