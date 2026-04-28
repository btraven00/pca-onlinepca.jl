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
| `solver` | string | One of `halko`, `ccipca`, `orthiter`, `arnoldi`, `algorithm971` |
| `n_components` | int | Number of PCs computed |
| `random_seed` | int | Random seed passed to the solver |

## Preprocessing

Genes are centered and scaled (per-gene zero-mean, unit-variance) before PCA. The input matrix is subsetted to the selected genes before scaling. OnlinePCA.jl is then invoked with `scale="raw"` so it does not reapply transformations on top of the already-scaled values.

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
