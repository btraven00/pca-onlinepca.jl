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
julia --project="$HERE" -e 'using Pkg; Pkg.instantiate()' >&2

exec julia --project="$HERE" -t "$NTHREADS" "$HERE/pca.jl" "$@"
