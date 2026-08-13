# OnlinePCA.jl

OnlinePCA.jl-backed PCA module for omnibenchmark scRNA pipelines.

Mirrors the layout and argument conventions of the sibling [`scanpy`](../scanpy) module so the two are interchangeable in an omnibenchmark stage. Output uses the shared neutral-HDF5 spec (`tool="OnlinePCA.jl"`).

Upstream: <https://github.com/rikenbit/OnlinePCA.jl>

## Setup

```sh
pixi install
pixi run instantiate
pixi run check
```

`pixi install` provisions Julia + Python (for the validator). `pixi run instantiate` resolves the Julia package deps from `Project.toml`. `pixi run check` imports all runtime Julia dependencies and prints `OK`.

## Usage

### PCA

```sh
pixi run julia --project=. pca.jl \
  --output_dir <dir> \
  --name <name> \
  --rawdata.h5ad <rawdata.h5ad> \
  --filtered.cellids <filtered.cellids.gz> \
  --selected.genes <selected.genes.gz> \
  --solver <tenxpca_sqrt|tenxpca_log|tenxpca_raw> \
  --n_components <int> \
  --random_seed <int>
```

Input is the **raw count h5ad** from the upstream `datasets` stage (read from `/layers/counts`), filtered to `--filtered.cellids` and subsetted to `--selected.genes` in this module. OnlinePCA.jl is integer-counts-only by design (memory-efficient streaming SVD); see [`docs/pca_output.md`](docs/pca_output.md) for the rationale and how this differs from the scanpy / scrapper PCA modules.

Output: `<output_dir>/<name>_<solver>_n_<n_components>.h5` — see [`docs/pca_output.md`](docs/pca_output.md) for the full format spec.

## Cell dropout

After subsetting raw counts to `--selected.genes`, some kept cells may have **zero total counts across the selected gene set**. OnlinePCA.jl's `tenxnormalizex` divides by per-cell totals for `scale ∈ {sqrt, log}`, so empty cells produce `NaN`/`Inf` and crash the LU step. This module drops such cells before running the SVD.

Consequence: this module's output `/cell_ids` may be a strict subset of the cell set used by the sibling scanpy / scrapper PCA modules (which scale per-gene rather than per-cell and so don't have this failure mode). Downstream code that joins PCA outputs across modules should treat each module's `/cell_ids` as authoritative — don't assume a fixed cell roster from the upstream filter stage.

## Embedding scale convention

OnlinePCA.jl defines `Scores[:, n] = λ[n] * V[:, n]` where `λ = σ² / M` (eigenvalues of the cell-side covariance), so its `/embedding` values are scaled by `σ/M` relative to the standard PCA convention `σ * V` used by scanpy / sklearn. Loadings (`/loadings = U`) and per-PC variance (`/variance = λ`, `/variance_ratio = λ / TotalVar`) follow the standard convention.

In practice this means: scanpy and OnlinePCA.jl embeddings are **proportional per PC but not numerically equal**. Most downstream metrics (clustering, ARI, k-NN graphs, cosine distances) are scale-invariant per-axis and unaffected. Anything that compares raw embedding distances across modules — direct Euclidean distance, RMSE between embeddings — will see a non-trivial difference and should normalize first (e.g. column-rescale to unit variance, or compare loadings instead).

### Procrustes-style cross-module comparisons

Vanilla orthogonal Procrustes (and its uniform-scale variant) cannot absorb the per-PC scale mismatch — each component has its own `λ_k` factor, not a single global scalar. Comparing embeddings across modules with Procrustes therefore needs one of:

1. **Column-standardize embeddings first**, then orthogonal Procrustes. Z-score each PC (zero mean, unit variance per column) before alignment. This is the recommended recipe for cross-implementation PCA benchmarks.
2. **Compare loadings instead of scores**. `/loadings = U` are unit-norm columns regardless of the σ-vs-λ scaling, so they're directly comparable up to sign across all PCA modules in this benchmark.
3. **Per-component scaled Procrustes** (rotation + diagonal scaling matrix `diag(s_k)`). Recovers the per-PC correction factor explicitly; the fitted `s_k` *is* the σ-vs-λ ratio.

## Threading

`pca.sh` reads `JULIA_NUM_THREADS` (falling back to `SLURM_CPUS_PER_TASK`, then `1`) and passes it to `julia -t`. Once omnibenchmark grows an env-passthrough stanza, the benchmark config can set `JULIA_NUM_THREADS={threads}` and the value will follow snakemake's `threads:` automatically.

**TODO:** also align BLAS threads with `BLAS.set_num_threads(Threads.nthreads())` at the top of `main()` to avoid oversubscription. `tenxpca` is BLAS-bound (randomized SVD on chunked reads), so this matters.

This module consumes raw counts directly from the `datasets` stage rather than going through the `three-normalize` stage, because OnlinePCA.jl bundles normalization into the streaming PCA pass (`scale=sqrt|log|raw`). The benchmark wires this as a parallel stage `five-pca-counts` alongside `five-pca` (normalized-input scanpy/scrapper); both contribute to the same `pca` output id downstream.

## Citation

If you use this module in your research, please cite it using the information in `CITATION.cff`. Please also cite OnlinePCA.jl (Tsuyuzaki et al., 2020).
