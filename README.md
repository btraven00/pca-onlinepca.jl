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
  --normalized.h5 <normalized.h5> \
  --selected.genes <selected.genes.gz> \
  --solver <halko|ccipca|orthiter|arnoldi|algorithm971> \
  --n_components <int> \
  --random_seed <int>
```

Output: `<output_dir>/<name>_<solver>_n_<n_components>.h5` — see [`docs/pca_output.md`](docs/pca_output.md) for the full format spec.

#### Validation

```sh
pixi run validate <output_dir>/<name>_pca.h5
```

Exit codes: `0` = valid, `1` = validation failure, `2` = IO / usage error.

## Citation

If you use this module in your research, please cite it using the information in `CITATION.cff`. Please also cite OnlinePCA.jl (Tsuyuzaki et al., 2020).
