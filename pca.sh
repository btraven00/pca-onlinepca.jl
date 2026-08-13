#!/usr/bin/env bash
# Entrypoint wrapper: activates the module's Julia project, instantiates
# deps on first run (no-op afterwards), forwards args to pca.jl.
#
# Threading: honors JULIA_NUM_THREADS, falling back to SLURM_CPUS_PER_TASK,
# then to 1. omnibenchmark/snakemake should set JULIA_NUM_THREADS={threads}
# once the upstream env-passthrough lands; meanwhile we degrade to single-
# threaded so behaviour is deterministic.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NTHREADS="${JULIA_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Idempotent: Pkg.instantiate() is a no-op when the project is already
# resolved against an existing Manifest.toml.
#
# ponytail: serialized with flock -- parallel jobs share one depot and
# racing builds collide (WebIO's build.jl `cp`s bundles onto files a
# sibling already installed read-only -> EACCES). Julia's own pidfiles
# cover precompile but not Pkg build. Drop the lock once the depot is
# pre-warmed at env-build time instead of per-job.
DEPOT="${JULIA_DEPOT_PATH:-$HOME/.julia}"; DEPOT="${DEPOT%%:*}"
mkdir -p "$DEPOT"
flock "$DEPOT/.ob-instantiate.lock" \
  julia --project="$HERE" -e 'using Pkg; Pkg.instantiate()' >&2

exec julia --project="$HERE" -t "$NTHREADS" "$HERE/pca.jl" "$@"
