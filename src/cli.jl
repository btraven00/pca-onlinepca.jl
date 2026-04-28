"""
Argument parser for omnibenchmark OnlinePCA.jl modules.

Conventions match the sibling scanpy module:
- All arguments are required. No defaults — callers (omnibenchmark configs)
  must pass everything explicitly so runs are fully reproducible from the
  invocation line.
- ArgParse rejects unknown flags by default; we rely on that for strictness.
"""

module CLI

using ArgParse

export build_pca_parser

const VALID_SOLVERS = ["halko", "ccipca", "orthiter", "arnoldi", "algorithm971"]

function build_pca_parser()
    s = ArgParseSettings(description = "OmniBenchmark PCA module (OnlinePCA.jl)",
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
        "--normalized.h5"
            dest_name = "normalized_h5"
            help = "TENx-format HDF5 of normalized expression (genes x cells)"
            arg_type = String
            required = true
        "--selected.genes"
            dest_name = "selected_genes"
            help = "Gzipped text file of selected gene ids (one per line)"
            arg_type = String
            required = true
        "--solver"
            help = "PCA solver: " * join(VALID_SOLVERS, ", ")
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
