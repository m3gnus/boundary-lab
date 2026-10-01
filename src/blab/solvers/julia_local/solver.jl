"""
The BEAT Engine worker entry point.

Nothing but loading and dispatch lives here. The engine is in `src/`, the
driver is in `BeatEngineDriver.jl`, and both are `include`d by a bundle
package under `julia_engine/` so that Julia can cache their native code in a
pkgimage. A script cannot be cached -- it is parsed, lowered and compiled from
source in every process -- and this file used to hold the whole driver, which
made the worker's own entry path the largest single item in a cold start.

The fallback below is not a nicety. Analysis scripts, an un-instantiated
checkout, and any environment whose bundle has not been resolved all take it,
and they behave exactly as they did before the bundles existed: same code,
same results, just compiled from source again.

`BLAB_BEAT_ENGINE_BUNDLE=0` forces that fallback, and exists for one specific
job. A bundle's precompile workload compiles the numerical kernels ahead of
time, and LLVM's output for an ahead-of-time compilation is not bit-identical
to the JIT's: measured against the pre-bundle path on the CPU backend over
A1-A3 at 500, 2000 and 6000 Hz, SPL agrees to 1.6e-5 dB and pressures to about
one Float32 ulp. That is two orders of magnitude below the Metal backend's own
run-to-run atomics noise and far below anything acoustic, but it is not zero --
so when a number has to be reproduced exactly, this variable gives back the
old path, which *is* bit-identical, at the old start-up cost.
"""

# Match the Metal bundle's dependency order to avoid invalidating its cache.
let hint = lowercase(strip(get(ENV, "BLAB_BEAT_ENGINE_GPU_BACKEND", "")))
    active = Base.active_project()
    directory = active === nothing ? "" : lowercase(basename(dirname(active)))
    if hint == "metal" || (isempty(hint) && directory == "julia_metal")
        @eval import Metal
    end
end

using JSON
using LinearAlgebra
using Printf
using Statistics
using StaticArrays

#: The bundle package for the accelerator this process is configured for. The
#: environment variable wins because that is what the Python wrapper sets when
#: it starts a worker; the project directory is the fallback the CLI relies on.
const BEAT_ENGINE_BUNDLE_NAME = let
    hint = lowercase(strip(get(ENV, "BLAB_BEAT_ENGINE_GPU_BACKEND", "")))
    if isempty(hint)
        active = Base.active_project()
        directory = active === nothing ? "" : lowercase(basename(dirname(active)))
        hint = directory == "julia_cuda" ? "cuda" :
            directory == "julia_rocm" ? "rocm" :
            directory == "julia_metal" ? "metal" : "cpu"
    end
    hint == "cuda" ? :BeatEngineCudaBundle :
        hint == "rocm" ? :BeatEngineRocmBundle :
        hint == "metal" ? :BeatEngineMetalBundle : :BeatEngineCpuBundle
end

const BEAT_ENGINE_BUNDLE = if get(ENV, "BLAB_BEAT_ENGINE_BUNDLE", "1") == "0"
    nothing
else
    try
        @eval using $BEAT_ENGINE_BUNDLE_NAME
        @eval $BEAT_ENGINE_BUNDLE_NAME
    catch
        nothing
    end
end

if BEAT_ENGINE_BUNDLE === nothing
    include(joinpath(@__DIR__, "src", "BeatEngineCore.jl"))
    @eval using .BeatEngineCore
    include(joinpath(@__DIR__, "BeatEngineDriver.jl"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if BEAT_ENGINE_BUNDLE === nothing
        main(ARGS)
    else
        BEAT_ENGINE_BUNDLE.main(ARGS)
    end
end
