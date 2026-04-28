#!/usr/bin/env julia
# test that all runtime dependencies import cleanly
using HDF5
using SparseArrays
using ArgParse
using GZip
using DelimitedFiles
using OnlinePCA

println("OK")
