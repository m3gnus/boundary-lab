# Refit the adaptive dense-solve cost model for this machine.
#
#   julia -t auto --project=<backend project> scripts/calibrate_dense_solve.jl [sizes...]
#
# The router in BeatEngineDenseSolve.jl chooses dense LU or GMRES by comparing
# two modelled costs, and the model carries four measured constants. They ship
# calibrated for an Apple M1 Max; any other machine should refit them, because
# what the model actually weighs is that machine's ratio of dense-GEMM
# throughput to memory bandwidth.
#
# This script measures all four on synthetic complex matrices of the same
# element type and shape the solver uses, and prints the export lines. It does
# not need a mesh: the primitives are BLAS, not BEM.
#
# Take at least three sizes. Two points cannot tell a wrong exponent from a
# wrong constant, and the linear term in the matvec model exists precisely
# because the effective bandwidth is still rising across the useful range.
#
# The reduction from samples to constants is deliberately not a mean. On the
# M1 Max every measured quantity is flat across the useful range, so any
# reduction agrees; on a machine where they are not, the choice decides whether
# the router misroutes. Measured on a Ryzen 7 5825U (2026-09-02): LU throughput
# ramps 93 to 420 GFLOP/s across 1,024 to 20,422 dofs, the triangular solve is
# 42 GB/s cache-resident at the bottom against 21-25 GB/s where routing
# decisions actually happen, and the matvec's effective bandwidth falls with N
# instead of rising. A mean LU rate, a maximum triangular rate and a free
# linear matvec term all describe that machine wrongly. What matters is
# accuracy near the crossover, because that is the only place the two costs are
# close, so the scalar constants are read off the crossover rather than
# averaged.

using LinearAlgebra, Printf, Random

include(joinpath(get(ENV, "BLAB_CORE_DIR", joinpath(@__DIR__, "..", "src")), "BeatEngineCore.jl"))
using .BeatEngineCore

const SIZES = isempty(ARGS) ? [2048, 5107, 10230, 20422] : sort(parse.(Int, ARGS))
const PASSES = parse(Int, get(ENV, "CALIBRATE_PASSES", "3"))
const ITERATION_SAMPLE = parse(Int, get(ENV, "CALIBRATE_MATVEC_ITERATIONS", "20"))

best(f) = minimum(begin f() end for _ in 1:PASSES)

"""
    warmup!(n=512)

Compile `lu!`, `mul!` and `ldiv!` before anything is timed. `best()` takes a
minimum over passes, which removes noise but not compilation: the first pass at
the first size pays it, and with the default three passes it can still dominate
that size's minimum. Left unwarmed the smallest size reads about 5x slow, and
the smallest size is the one nearest the crossover.
"""
function warmup!(n::Int=512)
    matrix = ComplexF32.(randn(Float32, n, n), randn(Float32, n, n))
    @inbounds for index in 1:n
        matrix[index, index] += ComplexF32(2 * sqrt(n))
    end
    x = ComplexF32.(randn(Float32, n), randn(Float32, n))
    y = similar(x)
    for _ in 1:3
        mul!(y, matrix, x)
        factorization = lu!(copy(matrix))
        ldiv!(factorization, copy(x))
    end
    return nothing
end

function measure(n::Int)
    Random.seed!(20260902 + n)
    matrix = ComplexF32.(randn(Float32, n, n), randn(Float32, n, n))
    # Make it non-singular and diagonally sane, as the assembled operator is.
    @inbounds for index in 1:n
        matrix[index, index] += ComplexF32(2 * sqrt(n))
    end
    x = ComplexF32.(randn(Float32, n), randn(Float32, n))
    y = similar(x)

    matvec_seconds = best() do
        @elapsed for _ in 1:ITERATION_SAMPLE
            mul!(y, matrix, x)
        end
    end / ITERATION_SAMPLE

    factorization = nothing
    lu_seconds = best() do
        work = copy(matrix)
        elapsed = @elapsed(factorization = lu!(work))
        elapsed
    end

    triangular_seconds = best() do
        rhs = copy(x)
        @elapsed(ldiv!(factorization, rhs))
    end

    lu_gflops = (8 / 3) * float(n)^3 / lu_seconds / 1e9
    matvec_gbps = 8 * float(n)^2 / matvec_seconds / 1e9
    triangular_gbps = 8 * float(n)^2 / triangular_seconds / 1e9
    return (; n, matvec_seconds, lu_seconds, triangular_seconds,
            lu_gflops, matvec_gbps, triangular_gbps)
end

"""
    fit_matvec(samples)

Least squares for `t = a N^2 + b N`, with `b` constrained non-negative. Returns
`(a, b, clamped)`.

`b` carries the sub-saturation ramp: on a machine whose matvec bandwidth rises
with N it is positive, and the model's local exponent comes out sub-quadratic
over the useful range, which is what the shipped M1 Max constants describe.
Where bandwidth falls with N instead -- a smaller cache and less memory
parallelism, so the stream leaves cache sooner -- the free fit returns `b < 0`,
and `_beat_env_float` rejects any non-positive value outright. The
unconstrained fit is therefore not merely inaccurate there: it prints an export
line the solver refuses to start with.

With the constraint active the optimum is `b = 0`. The data wants negative, so
the closest feasible point is the boundary and `a` is refitted alone.
"""
function fit_matvec(samples)
    design = reduce(vcat, [float(s.n)^2 float(s.n)] for s in samples)
    observed = [s.matvec_seconds for s in samples]
    coefficients = design \ observed
    entry, per_dof = coefficients[1], coefficients[2]
    per_dof >= 0 && return entry, per_dof, false
    squares = [float(s.n)^2 for s in samples]
    return sum(squares .* observed) / sum(squares .^ 2), 0.0, true
end

"""Linear interpolation of a per-size measurement, clamped outside the range."""
function interpolate(samples, field::Symbol, n::Real)
    length(samples) == 1 && return getfield(samples[1], field)
    n <= samples[1].n && return getfield(samples[1], field)
    n >= samples[end].n && return getfield(samples[end], field)
    index = findfirst(sample -> sample.n >= n, samples)
    low, high = samples[index - 1], samples[index]
    weight = (float(n) - low.n) / (high.n - low.n)
    return (1 - weight) * getfield(low, field) + weight * getfield(high, field)
end

function with_constants(body, lu_gflops, entry_seconds, dof_seconds, triangular_gbps, iterations)
    return withenv(
        body,
        BeatEngineCore.BEAT_DENSE_LU_GFLOPS_ENV => string(lu_gflops),
        BeatEngineCore.BEAT_DENSE_MATVEC_ENTRY_SECONDS_ENV => string(entry_seconds),
        BeatEngineCore.BEAT_DENSE_MATVEC_DOF_SECONDS_ENV => string(max(dof_seconds, 1e-30)),
        BeatEngineCore.BEAT_DENSE_TRIANGULAR_GBPS_ENV => string(triangular_gbps),
        BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_ENV => string(iterations),
    )
end

"""
    crossover_constants(samples, entry_seconds, dof_seconds, iterations, drives)

Read the two scalar constants off the crossover rather than averaging them.

The model's only job is to decide which of two costs is smaller, so it is only
ever wrong where they are close, and that is the crossover. A mean over the
sampled sizes weights a 2,048-dof point -- where the LU runs at a third of its
asymptotic rate and the triangular solve is still cache-resident -- exactly as
heavily as the size the decision is actually made at.

The crossover depends on the constants and the constants are read at the
crossover, so this is a fixed point. It converges in a few steps because both
measured curves are monotone and nearly flat in the neighbourhood; the
iteration is capped and returns its last iterate either way.
"""
function crossover_constants(samples, entry_seconds, dof_seconds, iterations, drives)
    lu_gflops = sum(sample.lu_gflops for sample in samples) / length(samples)
    triangular_gbps = samples[end].triangular_gbps
    crossover = samples[end].n
    for _ in 1:12
        previous = crossover
        crossover = with_constants(lu_gflops, entry_seconds, dof_seconds,
                                   triangular_gbps, iterations) do
            beat_dense_solve_crossover_dofs(drives)
        end
        lu_gflops = interpolate(samples, :lu_gflops, crossover)
        triangular_gbps = interpolate(samples, :triangular_gbps, crossover)
        abs(crossover - previous) <= 1 && break
    end
    return lu_gflops, triangular_gbps, crossover
end

warmup!()

println("BLAS threads: $(BLAS.get_num_threads())   Julia threads: $(Threads.nthreads())   passes: $PASSES")
if BLAS.get_num_threads() != Threads.nthreads()
    println()
    println("note: solver.jl sets BLAS threads to the Julia thread count. Calibrate at")
    println("      the setting the solver actually runs at, or these constants describe")
    println("      a configuration production never uses.")
end
println()
@printf("%8s  %12s  %10s  %12s  %10s  %12s  %10s\n",
        "N", "matvec s", "GB/s", "LU s", "GFLOP/s", "trisolve s", "GB/s")
samples = map(SIZES) do n
    sample = measure(n)
    @printf("%8d  %12.6f  %10.1f  %12.4f  %10.1f  %12.6f  %10.1f\n",
            sample.n, sample.matvec_seconds, sample.matvec_gbps,
            sample.lu_seconds, sample.lu_gflops,
            sample.triangular_seconds, sample.triangular_gbps)
    sample
end

entry_seconds, dof_seconds, clamped = fit_matvec(samples)

println()
if clamped
    println("Matvec bandwidth falls with N here, so the free fit of the linear term is")
    println("negative and the solver would reject it. Refitted with that term clamped")
    println("to zero; the quadratic term alone carries the model.")
    println()
end
println("Fitted matvec model: t = $(entry_seconds) N^2 + $(dof_seconds) N")
for sample in samples
    modelled = entry_seconds * float(sample.n)^2 + dof_seconds * float(sample.n)
    @printf("  N=%-8d measured %.6f s   model %.6f s   %.2fx\n",
            sample.n, sample.matvec_seconds, modelled, modelled / sample.matvec_seconds)
end

iterations = BeatEngineCore._beat_env_float(
    BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_ENV,
    BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_DEFAULT,
)
lu_gflops, triangular_gbps, crossover =
    crossover_constants(samples, entry_seconds, dof_seconds, iterations, 1)

println()
@printf("LU throughput spans %.0f-%.0f GFLOP/s over the sampled range and reads\n",
        minimum(s.lu_gflops for s in samples), maximum(s.lu_gflops for s in samples))
@printf("%.1f GFLOP/s at the single-drive crossover (%d dofs). Triangular solve\n",
        lu_gflops, crossover)
@printf("spans %.0f-%.0f GB/s and reads %.1f GB/s there.\n",
        minimum(s.triangular_gbps for s in samples),
        maximum(s.triangular_gbps for s in samples), triangular_gbps)

println()
println("Export these to calibrate this machine:")
println()
@printf("export %s=%.1f\n", BeatEngineCore.BEAT_DENSE_LU_GFLOPS_ENV, lu_gflops)
@printf("export %s=%.3e\n", BeatEngineCore.BEAT_DENSE_MATVEC_ENTRY_SECONDS_ENV, entry_seconds)
@printf("export %s=%.3e\n", BeatEngineCore.BEAT_DENSE_MATVEC_DOF_SECONDS_ENV,
        max(dof_seconds, 1e-30))
@printf("export %s=%.1f\n", BeatEngineCore.BEAT_DENSE_TRIANGULAR_GBPS_ENV, triangular_gbps)
println()
println("The iteration constant ($(BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_ENV)) is a")
println("property of the operator rather than the machine and is not measured here.")
println("Read it off a real solve's reported iteration counts.")
println()
println("Crossover with the constants fitted above, at $(iterations) iterations:")
with_constants(lu_gflops, entry_seconds, dof_seconds, triangular_gbps, iterations) do
    for drives in 1:4
        @printf("  %d drive(s): %d dofs\n", drives, beat_dense_solve_crossover_dofs(drives))
    end
end
