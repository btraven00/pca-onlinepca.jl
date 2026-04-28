"""
Argument parser for omnibenchmark OnlinePCA.jl module.

Conventions:
- All arguments are required. No defaults — callers (omnibenchmark configs)
  must pass everything explicitly so runs are fully reproducible.
- ArgParse rejects unknown flags by default; we rely on that for strictness.

OnlinePCA.jl is integer-counts-only (its `loadchromium` reads
matrix/data into a Vector{Int64}). We therefore consume the raw count
matrix from the upstream `datasets` stage (h5ad `/layers/counts`),
apply filtering + gene selection, and feed integer counts into
`tenxpca`. The `--solver` value selects the on-the-fly scale variant.
"""

module CLI

using ArgParse

export build_pca_parser, VALID_SOLVERS

const VALID_SOLVERS = ["tenxpca_sqrt", "tenxpca_log", "tenxpca_raw"]

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
        "--rawdata.h5ad"
            dest_name = "rawdata_h5ad"
            help = "h5ad with /layers/counts (cells x genes, integer)"
            arg_type = String
            required = true
        "--filtered.cellids"
            dest_name = "filtered_cellids"
            help = "Gzipped text file of kept cell barcodes (one per line)"
            arg_type = String
            required = true
        "--selected.genes"
            dest_name = "selected_genes"
            help = "Gzipped text file of selected gene ids (one per line)"
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
