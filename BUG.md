# Known upstream bug — OnlinePCA.jl `tenxpca` kwargs form

## Summary

The keyword-args entrypoint `OnlinePCA.tenxpca(; ...)` is broken at the commit we currently pin (`bfb2f75bcead28795303140cb8cac14af07d334d`, HEAD of `master` as of 2026-04-29).

Calling it raises:

```
UndefVarError: `pca` not defined in `OnlinePCA`
```

## Root cause

`src/tenxpca.jl:43` calls the positional-args inner method and passes a local `pca` symbol:

```julia
out = tenxpca(tenxfile, outdir, scale, rowmeanlist, rowvarlist, colsumlist,
              dim, noversamples, niter, chunksize, logdir, pca, ...)
```

…but the kwargs wrapper never assigns `pca`. Every sibling solver (`halko.jl`, `ccipca.jl`, `arnoldi.jl`, etc.) has an analogous line that initializes the dispatch tag, e.g.:

```julia
pca = HALKO()
```

`tenxpca.jl` is missing the corresponding `pca = TENXPCA()`. The struct exists (`Utils.jl:17: struct TENXPCA end`) and `parse_commandline(::TENXPCA)` is wired up (`Utils.jl:309`); only the kwargs wrapper is missing the one-liner.

## Workaround in this repo

`pca.jl` bypasses the kwargs wrapper and calls the positional inner method directly with an explicit `OnlinePCA.TENXPCA()` instance. Search for `# NOTE: bypass the kwargs wrapper` to find the block.

## TODO

- [ ] File a bug report against <https://github.com/rikenbit/OnlinePCA.jl> referencing commit `bfb2f75`, line `src/tenxpca.jl:43`.
- [ ] Submit a one-line PR adding `pca = TENXPCA()` immediately before line 43, mirroring the pattern used in every sibling solver file.
- [ ] Once merged + tagged, bump the `[sources]` pin in `Project.toml` to the fixed commit and remove the workaround block in `pca.jl`.
