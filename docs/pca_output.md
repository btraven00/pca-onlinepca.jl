# PCA output format (version 1)

File: `{output_dir}/{name}_{solver}_n_{n_components}.h5`

Flat HDF5 — intentionally not h5ad — so R (`rhdf5` / `HDF5Array`) and Julia (`HDF5.jl`) can read it without an anndata dependency. This module produces the same neutral format as the sibling `scanpy` and `scrapper` PCA modules.

## Datasets

| Path | dtype | Shape | Description |
|---|---|---|---|
| `/embedding` | float64 | `(n_cells, n_components)` | PCA scores, cells-as-rows |
| `/loadings` | float64 | `(n_genes, n_components)` | Gene loadings (rotation / components), genes-as-rows |
| `/variance` | float64 | `(n_components,)` | Variance explained by each PC |
| `/variance_ratio` | float64 | `(n_components,)` | Fraction of total variance explained per PC |
| `/cell_ids` | UTF-8 string | `(n_cells,)` | Cell barcodes, aligned with rows of `/embedding` |
| `/gene_ids` | UTF-8 string | `(n_genes,)` | Gene identifiers (selected/HVG subset), aligned with rows of `/loadings` |

**Note for R consumers:** `/embedding` is cells-as-rows, matching `SingleCellExperiment::reducedDim` orientation.

## Root attributes

| Attribute | Type | Description |
|---|---|---|
| `format_version` | string | Always `"1"` for this spec |
| `tool` | string | `OnlinePCA.jl` for this module |
| `tool_version` | string | Version of OnlinePCA.jl |
| `solver` | string | One of `tenxpca_sqrt`, `tenxpca_log`, `tenxpca_raw` — the `tenxpca` randomized SVD with the named on-the-fly scale variant applied to the raw count matrix |
| `n_components` | int | Number of PCs computed |
| `random_seed` | int | Random seed passed to the solver |

## Preprocessing

Input is the raw count matrix from the `datasets` stage (`/layers/counts` in the h5ad), filtered to `filtered.cellids` and subsetted to `selected.genes` in this module. The integer counts are written to a temporary TENx-format HDF5; `tenxsumr` computes per-row means and `tenxpca` performs randomized SVD with the chosen `scale` variant (`sqrt`, `log`, `raw`). Per-row centering uses the corresponding mean file (`Feature_SqrtMeans.csv`, `Feature_LogMeans.csv`, or `Feature_Means.csv`).

This differs from the scanpy / scrapper PCA modules, which consume the already-normalized matrix from the `three-normalize` stage. OnlinePCA.jl is integer-counts-only by design (its `loadchromium` reads matrix/data into a `Vector{Int64}` for memory-efficient OOC streaming), and bundles normalization into the streaming PCA pass via the `scale` argument.

## Validation

```bash
python validators/pca_output.py path/to/name_pca.h5
```

Exit codes: `0` = valid, `1` = validation failure, `2` = IO / usage error.

## Reading in Julia

```julia
using HDF5

h5open("name_pca.h5", "r") do h5
    embedding      = read(h5, "embedding")        # (n_cells, n_components)
    loadings       = read(h5, "loadings")         # (n_genes, n_components)
    variance       = read(h5, "variance")
    variance_ratio = read(h5, "variance_ratio")
    cell_ids       = read(h5, "cell_ids")
    gene_ids       = read(h5, "gene_ids")
    tool           = attrs(h5)["tool"]
    solver         = attrs(h5)["solver"]
end
```

## Reading in R

```r
library(rhdf5)

h5  <- H5Fopen("name_pca.h5")
emb <- h5read("name_pca.h5", "embedding")
lod <- h5read("name_pca.h5", "loadings")
var <- h5read("name_pca.h5", "variance")
vr  <- h5read("name_pca.h5", "variance_ratio")
cells <- h5read("name_pca.h5", "cell_ids")
genes <- h5read("name_pca.h5", "gene_ids")
H5Fclose(h5)
```
