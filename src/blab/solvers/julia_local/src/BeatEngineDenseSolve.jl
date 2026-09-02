# Adaptive dense solve for the fused Burton-Miller system.
#
# The fused exterior assembly hands over one N x N system matrix and an
# N x drives right-hand side. There are two ways to solve that, and which one
# is faster depends on the problem, not on a preference:
#
# * Dense LU. Cost (8/3)N^3 for the factorization, then one triangular solve
#   per drive. The factorization is shared by every drive, which is a property
#   the CPU and Metal backends were deliberately built to have.
# * GMRES with diagonal preconditioning. Cost iterations * one dense matvec,
#   per drive, with no factorization to share.
#
# Both are measured on the ATH ladder (see the head-to-head document). GMRES
# needs 35-89 iterations on this operator, so it is O(N^2) against the LU's
# O(N^3) and the two cross -- between 2,000 and 5,000 P1 dofs at one drive.
# In the drive count the ranking is the other way round, because the LU
# amortises its factorization across every drive and GMRES does not, and that
# drive crossover itself grows with N (2.4-2.7 drives at 5,107 dofs, 4.6-7.4 at
# 10,230). There is therefore no pair of thresholds to route on.
#
# An earlier revision of this file quoted "206-246 iterations, flat in N". That
# band was a Float32 Arnoldi recurrence losing orthogonality, not the operator;
# see the GMRES section below, which is built so it cannot recur.
#
# So the router is two-dimensional, over (N, drives), and it is a cost model
# rather than a dof threshold. A hardcoded threshold would be wrong on the
# CUDA and ROCm backends, which factor on the device, and wrong again on any
# machine whose ratio of dense-GEMM throughput to memory bandwidth differs
# from the one the constants below were calibrated on.
#
# Preconditioning: diagonal, and only diagonal. Block-Jacobi built from
# geometric leaf clustering -- the design hornlab-metal-bem uses -- was measured
# to shave a handful of iterations for several seconds of setup, and
# mass-matrix preconditioning was worse than nothing. Neither is available here
# on purpose. Those comparisons were made against the stalled Float32
# recurrence, so their *ratios* are not trustworthy; what survives is that the
# setup cost of a block preconditioner is large and fixed while the iteration
# count it has to beat is now 35-89, which makes the case against it stronger
# rather than weaker.
#
# GMRES falls back to the dense LU when it does not converge, reporting the
# fallback rather than raising. BEAT's Burton-Miller operator does not stagnate
# on the meshes tried, including the sliver-rim meshes where
# hornlab-metal-bem's GMRES returns `info=-999`, so this is a safety net rather
# than a load-bearing part of the feature. It earned its place during
# development regardless: the Float32 Arnoldi bug made a 10,230-dof solve fail
# to converge at 2 kHz, and the fallback turned that into a correct answer at
# the LU's price instead of a failed solve.

const BEAT_DENSE_SOLVE_METHOD_ENV = "BLAB_BEAT_DENSE_SOLVE"
const BEAT_GMRES_TOLERANCE_ENV = "BLAB_BEAT_GMRES_TOL"
const BEAT_GMRES_MAX_ITERATIONS_ENV = "BLAB_BEAT_GMRES_MAX_ITERATIONS"
const BEAT_GMRES_RESTART_ENV = "BLAB_BEAT_GMRES_RESTART"

# --- Cost-model calibration -------------------------------------------------
#
# Calibrated on an Apple M1 Max (10 cores, 64 GB) against the ATH reference
# ladder at 1,974 / 5,107 / 10,230 / 20,422 P1 dofs, Float32 complex, OpenBLAS
# ILP64. Every constant is an environment override so a different machine can
# be refitted without a code change; `scripts/calibrate_dense_solve.jl`
# measures all four and prints the export lines.
#
# These are the M1 Max numbers and nothing else. In particular they do not
# describe the CUDA or ROCm backends, whose dense LU runs on the device
# (`cuda_dense_lu!`, `rocm_dense_lu!`) and whose arithmetic-to-bandwidth ratio
# is not Metal's. Those backends do not route through here yet.

"""Effective complex GEMM throughput of the dense LU, GFLOP/s.

Measured 457-511 GFLOP/s over 2,048 to 20,422 dofs with no degradation at the
top (508 GFLOP/s at 20,422), so one constant fits the whole range."""
const BEAT_DENSE_LU_GFLOPS_ENV = "BLAB_BEAT_LU_GFLOPS"
const BEAT_DENSE_LU_GFLOPS_DEFAULT = 480.0

"""Streaming cost of one dense complex matvec, seconds per matrix entry.

8 bytes per Complex{Float32} entry against the asymptotic bandwidth a threaded
`cgemv` approaches here. Fitted, with the term below, to measured matvecs at
2,048 / 5,107 / 10,230 dofs."""
const BEAT_DENSE_MATVEC_ENTRY_SECONDS_ENV = "BLAB_BEAT_MATVEC_ENTRY_SECONDS"
const BEAT_DENSE_MATVEC_ENTRY_SECONDS_DEFAULT = 1.071e-10

"""Non-streaming part of one dense complex matvec, seconds per dof.

The matvec does not run at its asymptotic bandwidth on a small matrix: the
measured effective rate rises from 39.7 GB/s at 5,107 dofs to 67.7 GB/s at
20,422 as the transfer saturates. This linear term is what reproduces that
ramp, and it is why the model's local exponent comes out sub-quadratic over the
useful range instead of being forced to 2. Fitting
`iterations * 8N^2 / bandwidth` with a single bandwidth constant misplaces the
crossover at both ends."""
const BEAT_DENSE_MATVEC_DOF_SECONDS_ENV = "BLAB_BEAT_MATVEC_DOF_SECONDS"
const BEAT_DENSE_MATVEC_DOF_SECONDS_DEFAULT = 4.236e-7

"""Iterations a diagonally preconditioned GMRES is expected to need.

This is the model's only genuinely operator-dependent constant, and the one to
re-measure whenever the formulation, the coupling parameter or the quadrature
changes.

It is a compromise, not an invariant, and the docstring says so because the
number it replaced was presented as one. Measured on the ATH ladder at
tolerance 1e-6 on the true residual:

    mesh   N        500 Hz   2 kHz   6 kHz
    A5     5,107      70       51      51
    A2r   10,230     119       85      63
    A5r   20,422      --       79      --

Roughly flat in N and a factor of ~2 in frequency, harder at the bottom of the
band because the uncapped Burton-Miller coupling `eta = i/k` degrades
conditioning as k goes to zero. 70 is mid-range rather than worst case: it
misroutes only where the two costs are within a few percent of each other, and
being wrong there costs a few percent. A worst-case 119 would additionally send
genuine GMRES wins at 5,107 dofs to the LU."""
const BEAT_GMRES_MODEL_ITERATIONS_ENV = "BLAB_BEAT_GMRES_MODEL_ITERATIONS"
const BEAT_GMRES_MODEL_ITERATIONS_DEFAULT = 70.0

"""Effective bandwidth of one triangular solve against the LU factor, GB/s.

This is what an extra drive costs on the LU path, and it is not the matvec
rate: a triangular solve is sequential in the row dimension and measures
13-17 GB/s where `cgemv` reaches 42-55. Its own constant, because the LU-side
drive term is what the router weighs GMRES against."""
const BEAT_DENSE_TRIANGULAR_GBPS_ENV = "BLAB_BEAT_TRIANGULAR_GBPS"
const BEAT_DENSE_TRIANGULAR_GBPS_DEFAULT = 17.0

function _beat_env_float(name::AbstractString, default::Float64)
    text = strip(get(ENV, name, ""))
    isempty(text) && return default
    value = tryparse(Float64, text)
    value === nothing && error("$name must be a number; got $(repr(text)).")
    value > 0 || error("$name must be positive; got $value.")
    return value
end

function _beat_env_int(name::AbstractString, default::Int; allow_zero::Bool=false)
    text = strip(get(ENV, name, ""))
    isempty(text) && return default
    value = tryparse(Int, text)
    value === nothing && error("$name must be an integer; got $(repr(text)).")
    (value > 0 || (allow_zero && value == 0)) ||
        error("$name must be a positive integer; got $value.")
    return value
end

"""
    beat_dense_lu_seconds(n, drive_count)

Modelled dense-LU solve seconds: one `(8/3)N^3` factorization shared by every
drive, plus one triangular solve per drive. The triangular solve is bandwidth
bound on the factor rather than arithmetic bound, so it is priced as one pass
over the matrix -- the same cost as one matvec.
"""
function beat_dense_lu_seconds(n::Integer, drive_count::Integer=1)
    n <= 0 && return 0.0
    gflops = _beat_env_float(BEAT_DENSE_LU_GFLOPS_ENV, BEAT_DENSE_LU_GFLOPS_DEFAULT)
    factorization = (8 / 3) * float(n)^3 / (gflops * 1e9)
    return factorization + max(0, drive_count) * beat_dense_triangular_seconds(n)
end

"""
    beat_dense_triangular_seconds(n)

Modelled seconds for the pair of triangular solves one extra drive costs on
the LU path: one pass over the `N x N` factor at its own measured rate.
"""
function beat_dense_triangular_seconds(n::Integer)
    n <= 0 && return 0.0
    gbps = _beat_env_float(BEAT_DENSE_TRIANGULAR_GBPS_ENV, BEAT_DENSE_TRIANGULAR_GBPS_DEFAULT)
    return 8 * float(n)^2 / (gbps * 1e9)
end

"""
    beat_dense_matvec_seconds(n)

Modelled seconds for one dense `N x N` complex matvec, `a N^2 + b N`. The
quadratic term is the matrix stream; the linear term is what keeps the model
honest below saturation.
"""
function beat_dense_matvec_seconds(n::Integer)
    n <= 0 && return 0.0
    entry = _beat_env_float(BEAT_DENSE_MATVEC_ENTRY_SECONDS_ENV, BEAT_DENSE_MATVEC_ENTRY_SECONDS_DEFAULT)
    per_dof = _beat_env_float(BEAT_DENSE_MATVEC_DOF_SECONDS_ENV, BEAT_DENSE_MATVEC_DOF_SECONDS_DEFAULT)
    return entry * float(n)^2 + per_dof * float(n)
end

"""
    beat_gmres_seconds(n, drive_count)

Modelled GMRES seconds: one matvec per iteration, every iteration, for every
drive. Arnoldi orthogonalization is deliberately not modelled -- at 210
iterations and 20,422 dofs the Krylov basis is 34 MB against a 3.3 GB matrix,
so it is under 1% of the traffic.
"""
function beat_gmres_seconds(n::Integer, drive_count::Integer=1)
    n <= 0 && return 0.0
    iterations = _beat_env_float(BEAT_GMRES_MODEL_ITERATIONS_ENV, BEAT_GMRES_MODEL_ITERATIONS_DEFAULT)
    return max(1, drive_count) * iterations * beat_dense_matvec_seconds(n)
end

"""How much LU-equivalent work a model-chosen GMRES may spend before it stops.

One is the value that makes the argument: at a budget of one LU, a GMRES that
exhausts it and falls back has cost one LU of matvecs plus the factorization it
should have done, so the worst case is bounded at twice the direct solve. The
knob exists because that bound trades against false abandons -- a run that
would have converged a few iterations past the budget is stopped at 2x when it
would have finished near 1.2x -- and the balance is a property of how heavy the
operator's tail is, which differs by mesh family."""
const BEAT_GMRES_BUDGET_FACTOR_ENV = "BLAB_BEAT_GMRES_BUDGET"
const BEAT_GMRES_BUDGET_FACTOR_DEFAULT = 1.0

"""
    beat_gmres_iteration_budget(n, drive_count)

Total matvecs a model-chosen GMRES may spend across every drive before the
dense LU is provably the cheaper answer.

`BEAT_GMRES_MODEL_ITERATIONS` is the model's weakest constant by a wide margin.
It ships at 70; measured on this branch over three meshes and three frequencies
the true count runs from 39 to 549, and it is not monotone in either dofs or
frequency, so no single value is right. The router therefore cannot avoid
choosing GMRES on operators that need six times what it assumed, and with the
iteration cap set to `min(n, 1000)` the resulting loss is unbounded: at 4,751
dofs and 6 kHz a GMRES the router picked spent 429 iterations to beat a 0.88 s
factorization in 3.40 s.

The budget converts that into a bounded one. Exceeding it is not a failure to
be retried but the discovery that the routing decision was wrong, and the
correct response to that discovery is the LU. It applies only when the model
chose GMRES: an explicit `BLAB_BEAT_DENSE_SOLVE=gmres` is an instruction, and
benchmarking the Krylov path against the direct one needs it to run to the end.
"""
function beat_gmres_iteration_budget(n::Integer, drive_count::Integer=1)
    n <= 0 && return typemax(Int)
    matvec = beat_dense_matvec_seconds(n)
    matvec > 0 || return typemax(Int)
    factor = _beat_env_float(BEAT_GMRES_BUDGET_FACTOR_ENV, BEAT_GMRES_BUDGET_FACTOR_DEFAULT)
    return max(1, floor(Int, factor * beat_dense_lu_seconds(n, drive_count) / matvec))
end

"""
    beat_dense_solve_crossover_dofs(drive_count=1)

The dof count at which the model's two costs are equal, for a given number of
drives. Reported for documentation and tests; the router compares the two
costs directly and never needs this value.
"""
function beat_dense_solve_crossover_dofs(drive_count::Integer=1)
    low, high = 1, 1 << 24
    beat_gmres_seconds(high, drive_count) < beat_dense_lu_seconds(high, drive_count) || return high
    beat_gmres_seconds(low, drive_count) > beat_dense_lu_seconds(low, drive_count) || return low
    while high - low > 1
        middle = (low + high) >> 1
        if beat_gmres_seconds(middle, drive_count) < beat_dense_lu_seconds(middle, drive_count)
            high = middle
        else
            low = middle
        end
    end
    return high
end

"""
    beat_dense_solve_method(; override=ENV)

`:lu`, `:gmres` or `:auto` from `BLAB_BEAT_DENSE_SOLVE`.
"""
function beat_dense_solve_method(override::AbstractString=get(ENV, BEAT_DENSE_SOLVE_METHOD_ENV, "auto"))
    text = lowercase(strip(override))
    (isempty(text) || text == "auto") && return :auto
    text in ("lu", "dense", "direct") && return :lu
    text in ("gmres", "krylov", "iterative") && return :gmres
    error("$BEAT_DENSE_SOLVE_METHOD_ENV must be auto, lu or gmres; got $(repr(override)).")
end

"""
    beat_dense_solve_plan(n, drive_count; method=beat_dense_solve_method())

Which solver to use and why. Returns a named tuple carrying the chosen
`method`, both modelled costs, and the `reason` -- `:override` when the choice
was forced, `:model` when the cost model chose it.
"""
function beat_dense_solve_plan(n::Integer, drive_count::Integer=1;
                               method::Symbol=beat_dense_solve_method())
    lu_seconds = beat_dense_lu_seconds(n, drive_count)
    gmres_seconds = beat_gmres_seconds(n, drive_count)
    if method === :auto
        chosen = gmres_seconds < lu_seconds ? :gmres : :lu
        reason = :model
    else
        chosen = method
        reason = :override
    end
    return (
        method=chosen,
        reason=reason,
        dofs=Int(n),
        drives=Int(drive_count),
        lu_model_seconds=lu_seconds,
        gmres_model_seconds=gmres_seconds,
    )
end

# --- GMRES ------------------------------------------------------------------
#
# Full (unrestarted) GMRES with left diagonal preconditioning, modified
# Gram-Schmidt with selective reorthogonalization, and a Krylov space carried
# in Float64 while the operator stays Float32.
#
# The precision of the Krylov space is not a detail. The first version of this
# file kept the basis in Complex{Float32} to match the matrix, with DGKS
# reorthogonalization to protect it, and it needed 584-1000 iterations at
# 10,230 dofs where the true count is 35-42. Two remedies were tried and agree
# with each other -- a Float64 Krylov space, and unconditional
# reorthogonalization of a Float32 one -- which is the evidence that the
# excess was lost orthogonality and not the operator's conditioning.
#
# The cost of the Float64 basis is one conversion per vector per iteration,
# O(N) against the matvec's O(N^2), and 16 bytes per entry for a basis that is
# 42 vectors deep: 13 MB at 20,422 dofs, against a 3.3 GB matrix. There is no
# reason to run the space in Float32, and `BLAB_BEAT_GMRES_KRYLOV_PRECISION=f32`
# exists only so the failure can be reproduced on demand.
#
# `BLAB_BEAT_GMRES_REORTHOGONALIZE` selects `dgks` (default; reorthogonalize
# when the vector loses most of its norm), `always`, or `never`.
#
# Unrestarted is the right default: at 42 iterations the basis is a rounding
# error beside the matrix, and restarting only costs iterations. That is also
# the diagnostic that identified the bug -- a converging GMRES does not care
# what the restart length is, so an iteration count that tracks it is not an
# iteration count.

const BEAT_GMRES_KRYLOV_PRECISION_ENV = "BLAB_BEAT_GMRES_KRYLOV_PRECISION"
const BEAT_GMRES_REORTHOGONALIZE_ENV = "BLAB_BEAT_GMRES_REORTHOGONALIZE"

"""Fraction of the target the inner Givens recursion aims at.

The recursion measures the preconditioned residual and the tolerance is on the
true one, so the inner cycle aims well past the target and the outer loop
verifies. The margin is not free to choose loosely: at 0.2 a 10,230-dof solve
handed back at a true residual of 1.02e-6 against a 1e-6 tolerance, the next
cycle improved by too little to clear the stall check, and the whole Krylov
solve was discarded for a dense LU over a 2% miss. Overshooting costs a few
iterations; missing costs the entire factorization."""
const BEAT_GMRES_INNER_MARGIN = 0.2

"""DGKS threshold: reorthogonalize when a vector loses this much of its norm."""
const BEAT_GMRES_DGKS_THRESHOLD = 0.7

"""A restart cycle must cut the true residual by at least this factor.

Otherwise the run is stalled and further cycles are wasted. This matters most
when the tolerance is simply unreachable: in Float32 the achievable true
relative residual floors out near `sqrt(N) * eps`, so a tolerance below that
would otherwise consume the whole iteration budget before falling back."""
const BEAT_GMRES_STAGNATION_FACTOR = 0.9

struct BeatGmresResult{T<:AbstractFloat}
    converged::Bool
    iterations::Int
    relative_residual::T
end

function beat_gmres_krylov_type(override::AbstractString=get(ENV, BEAT_GMRES_KRYLOV_PRECISION_ENV, "f64"))
    text = lowercase(strip(override))
    (isempty(text) || text in ("f64", "float64", "double")) && return ComplexF64
    text in ("f32", "float32", "single") && return ComplexF32
    error("$BEAT_GMRES_KRYLOV_PRECISION_ENV must be f64 or f32; got $(repr(override)).")
end

function beat_gmres_reorthogonalization(override::AbstractString=get(ENV, BEAT_GMRES_REORTHOGONALIZE_ENV, "dgks"))
    text = lowercase(strip(override))
    (isempty(text) || text == "dgks") && return :dgks
    text == "always" && return :always
    text == "never" && return :never
    error("$BEAT_GMRES_REORTHOGONALIZE_ENV must be dgks, always or never; got $(repr(override)).")
end

"""
    beat_gmres!(x, matrix, b; tolerance, max_iterations, restart, preconditioner)

Solve `matrix * x = b` in place. Left diagonal preconditioning is applied
internally; `tolerance` is on the true relative residual `||b - A x|| / ||b||`,
which is verified against the operator rather than trusted from the Givens
recursion.

Returns a `BeatGmresResult`. A non-converged result is a fact to act on, not
an error: the caller falls back to the dense LU.
"""
function beat_gmres!(x::AbstractVector{Complex{T}},
                     matrix::AbstractMatrix{Complex{T}},
                     b::AbstractVector{Complex{T}};
                     tolerance::Real=_beat_gmres_tolerance(T),
                     max_iterations::Integer=_beat_gmres_max_iterations(size(matrix, 1)),
                     restart::Integer=_beat_gmres_restart(),
                     krylov_type::Type=beat_gmres_krylov_type(),
                     reorthogonalize::Symbol=beat_gmres_reorthogonalization(),
                     preconditioner::Union{Nothing,AbstractVector{Complex{T}}}=nothing) where {T<:AbstractFloat}
    n = size(matrix, 1)
    size(matrix, 2) == n || error("beat_gmres! needs a square system; got $(size(matrix)).")
    length(b) == n || error("beat_gmres! right-hand side must have $n rows; got $(length(b)).")
    length(x) == n || error("beat_gmres! solution vector must have $n rows; got $(length(x)).")

    W = krylov_type
    tol = T(tolerance)
    inverse_diagonal = preconditioner === nothing ? beat_diagonal_preconditioner(matrix) : preconditioner
    cycle_length = restart <= 0 ? Int(max_iterations) : min(Int(restart), Int(max_iterations))

    b_norm = norm(b)
    if b_norm == 0
        fill!(x, zero(Complex{T}))
        return BeatGmresResult{T}(true, 0, zero(T))
    end
    preconditioned_b_norm = norm(b .* inverse_diagonal)
    preconditioned_b_norm == 0 && (preconditioned_b_norm = b_norm)

    residual = Vector{Complex{T}}(undef, n)   # operator precision
    applied = Vector{Complex{T}}(undef, n)    # matvec output, operator precision
    work = Vector{W}(undef, n)                # Krylov precision
    basis = Vector{Vector{W}}()
    total_iterations = 0
    previous_relative = T(Inf)

    while total_iterations < max_iterations
        # True residual of the current iterate. This is the quantity the
        # tolerance is on, and it is recomputed against the operator every
        # cycle rather than carried forward through the recursion.
        copyto!(residual, b)
        mul!(residual, matrix, x, -one(Complex{T}), one(Complex{T}))
        relative = T(norm(residual) / b_norm)
        relative <= tol && return BeatGmresResult{T}(true, total_iterations, relative)
        # A cycle that does not materially reduce the true residual will not
        # be rescued by another one. Without this, an unreachable tolerance --
        # and in Float32 the true residual floors out somewhere near
        # sqrt(N) * eps -- burns the entire iteration budget before the caller
        # gets its answer from the LU. Report the stall promptly instead.
        if relative > previous_relative * T(BEAT_GMRES_STAGNATION_FACTOR)
            return BeatGmresResult{T}(false, total_iterations, relative)
        end
        previous_relative = relative

        residual .*= inverse_diagonal
        isempty(basis) && push!(basis, Vector{W}(undef, n))
        @inbounds for index in 1:n
            work[index] = W(residual[index])
        end
        beta = norm(work)
        beta == 0 && return BeatGmresResult{T}(true, total_iterations, relative)
        basis[1] .= work ./ beta

        # The Krylov space cannot exceed the dimension of the problem, so
        # neither can a cycle. Without this cap an unrestarted run on a small
        # system keeps appending basis vectors to an exhausted space and
        # reports iteration counts larger than N.
        inner = min(cycle_length, max_iterations - total_iterations, n)
        inner_target = Float64(tol) * Float64(preconditioned_b_norm) * BEAT_GMRES_INNER_MARGIN
        # Happy breakdown: the subdiagonal never reaches exactly zero in
        # floating point, so it is tested against the scale of the recurrence.
        breakdown_floor = Float64(beta) * eps(Float64) * n

        hessenberg = zeros(ComplexF64, inner + 1, inner)
        givens_cosine = zeros(Float64, inner)
        givens_sine = zeros(ComplexF64, inner)
        rhs_small = zeros(ComplexF64, inner + 1)
        rhs_small[1] = ComplexF64(beta)
        used = 0

        for j in 1:inner
            # The basis grows one vector per iteration rather than being sized
            # for the iteration cap: unrestarted GMRES may be allowed 1,000
            # iterations and take 40.
            length(basis) < j + 1 && push!(basis, Vector{W}(undef, n))
            _beat_apply_operator!(applied, work, matrix, basis[j], inverse_diagonal)
            # The Arnoldi subdiagonal has to be read here. The Givens rotation
            # below zeroes `hessenberg[j + 1, j]` by construction -- that is
            # what it is for -- so a breakdown test applied after it fires on
            # every iteration, silently turning unrestarted GMRES into a
            # restart-of-one. That defect produced iteration counts of 103 to
            # 1,000 on meshes that need 35 to 60, and it looked exactly like a
            # conditioning problem.
            subdiagonal = _beat_arnoldi_step!(work, basis, hessenberg, j, reorthogonalize)
            _beat_apply_previous_givens!(hessenberg, givens_cosine, givens_sine, j)
            _beat_form_givens!(hessenberg, givens_cosine, givens_sine, rhs_small, j)
            used = j
            total_iterations += 1
            abs(rhs_small[j + 1]) <= inner_target && break
            subdiagonal <= breakdown_floor && break
        end

        _beat_gmres_update!(x, basis, hessenberg, rhs_small, used)
    end

    copyto!(residual, b)
    mul!(residual, matrix, x, -one(Complex{T}), one(Complex{T}))
    relative = T(norm(residual) / b_norm)
    return BeatGmresResult{T}(relative <= tol, total_iterations, relative)
end

# M^-1 A v, with the matvec at the operator's precision and the result handed
# back at the Krylov space's. The two conversions are O(N) against an O(N^2)
# matvec, which is what makes a Float64 Krylov space over a Float32 operator
# essentially free.
function _beat_apply_operator!(applied::Vector{Complex{T}},
                               work::Vector{W},
                               matrix::AbstractMatrix{Complex{T}},
                               vector::Vector{W},
                               inverse_diagonal::AbstractVector{Complex{T}}) where {T<:AbstractFloat,W}
    n = length(applied)
    if W === Complex{T}
        mul!(applied, matrix, vector)
    else
        source = Vector{Complex{T}}(undef, n)
        @inbounds for index in 1:n
            source[index] = Complex{T}(vector[index])
        end
        mul!(applied, matrix, source)
    end
    @inbounds for index in 1:n
        work[index] = W(applied[index] * inverse_diagonal[index])
    end
    return nothing
end

"""
    beat_diagonal_preconditioner(matrix)

`1 ./ diag(A)`, with a zero or non-finite diagonal entry left as 1 so the
preconditioner can never introduce a NaN into an otherwise solvable system.
"""
function beat_diagonal_preconditioner(matrix::AbstractMatrix{Complex{T}}) where {T<:AbstractFloat}
    n = min(size(matrix, 1), size(matrix, 2))
    inverse = Vector{Complex{T}}(undef, n)
    @inbounds for index in 1:n
        value = matrix[index, index]
        inverse[index] = (value == 0 || !isfinite(value)) ? one(Complex{T}) : inv(value)
    end
    return inverse
end

# Modified Gram-Schmidt. `dgks` reorthogonalizes when the vector loses most of
# its norm, which is enough for a Float64 space; `always` is what makes a
# Float32 space usable and is kept so the two remedies can be compared.
function _beat_arnoldi_step!(work::Vector{W},
                             basis::Vector{Vector{W}},
                             hessenberg::Matrix{ComplexF64},
                             j::Int,
                             reorthogonalize::Symbol) where {W}
    initial_norm = norm(work)
    @inbounds for i in 1:j
        coefficient = dot(basis[i], work)
        hessenberg[i, j] = ComplexF64(coefficient)
        axpy!(-coefficient, basis[i], work)
    end
    current_norm = norm(work)
    repeat = reorthogonalize === :always ||
        (reorthogonalize === :dgks && current_norm < BEAT_GMRES_DGKS_THRESHOLD * initial_norm)
    if repeat
        @inbounds for i in 1:j
            coefficient = dot(basis[i], work)
            hessenberg[i, j] += ComplexF64(coefficient)
            axpy!(-coefficient, basis[i], work)
        end
        current_norm = norm(work)
    end
    hessenberg[j + 1, j] = ComplexF64(current_norm)
    if current_norm > 0
        basis[j + 1] .= work ./ current_norm
    else
        fill!(basis[j + 1], zero(W))
    end
    # Returned rather than read back from the Hessenberg, which the Givens
    # rotation overwrites with zero a few lines later.
    return Float64(current_norm)
end

function _beat_apply_previous_givens!(hessenberg::Matrix{ComplexF64},
                                      givens_cosine::Vector{Float64},
                                      givens_sine::Vector{ComplexF64},
                                      j::Int)
    @inbounds for i in 1:(j - 1)
        upper = hessenberg[i, j]
        lower = hessenberg[i + 1, j]
        hessenberg[i, j] = givens_cosine[i] * upper + givens_sine[i] * lower
        hessenberg[i + 1, j] = -conj(givens_sine[i]) * upper + givens_cosine[i] * lower
    end
    return nothing
end

function _beat_form_givens!(hessenberg::Matrix{ComplexF64},
                            givens_cosine::Vector{Float64},
                            givens_sine::Vector{ComplexF64},
                            rhs_small::Vector{ComplexF64},
                            j::Int)
    @inbounds begin
        upper = hessenberg[j, j]
        lower = hessenberg[j + 1, j]
        magnitude = hypot(abs(upper), abs(lower))
        if magnitude == 0
            givens_cosine[j] = 1.0
            givens_sine[j] = 0.0 + 0.0im
            return nothing
        end
        givens_cosine[j] = abs(upper) / magnitude
        givens_sine[j] = upper == 0 ? ComplexF64(1) : (upper / abs(upper)) * conj(lower) / magnitude
        hessenberg[j, j] = givens_cosine[j] * upper + givens_sine[j] * lower
        hessenberg[j + 1, j] = 0.0 + 0.0im
        carried = rhs_small[j]
        rhs_small[j] = givens_cosine[j] * carried
        rhs_small[j + 1] = -conj(givens_sine[j]) * carried
    end
    return nothing
end

function _beat_gmres_update!(x::AbstractVector{Complex{T}},
                             basis::Vector{Vector{W}},
                             hessenberg::Matrix{ComplexF64},
                             rhs_small::Vector{ComplexF64},
                             used::Int) where {T<:AbstractFloat,W}
    used == 0 && return nothing
    y = Vector{ComplexF64}(undef, used)
    @inbounds for i in used:-1:1
        accumulated = rhs_small[i]
        for column in (i + 1):used
            accumulated -= hessenberg[i, column] * y[column]
        end
        pivot = hessenberg[i, i]
        y[i] = pivot == 0 ? ComplexF64(0) : accumulated / pivot
    end
    # Accumulate the update in the Krylov precision, not the solution's.
    #
    # x = x + sum_i y_i v_i over `used` terms. Summing that in Float32 carries a
    # relative error of about sqrt(used) * eps, which floors the achievable true
    # residual at roughly 1.1e-6 by 80 terms and 1.6e-6 by 175 -- and the floor
    # rises with the iteration count, so running longer makes the answer worse.
    # Measured at 10,230 dofs: 117 iterations reached 9.65e-7, 174 iterations
    # reached 1.33e-6. Accumulating in Float64 and rounding once at the end
    # removes the floor for the cost of one N-vector.
    accumulator = Vector{ComplexF64}(undef, length(x))
    @inbounds for index in eachindex(x)
        accumulator[index] = ComplexF64(x[index])
    end
    @inbounds for i in 1:used
        coefficient = y[i]
        vector = basis[i]
        for index in eachindex(accumulator)
            accumulator[index] += coefficient * ComplexF64(vector[index])
        end
    end
    @inbounds for index in eachindex(x)
        x[index] = Complex{T}(accumulator[index])
    end
    return nothing
end

function _beat_gmres_tolerance(::Type{T}) where {T<:AbstractFloat}
    return T(_beat_env_float(BEAT_GMRES_TOLERANCE_ENV, 1.0e-6))
end

function _beat_gmres_max_iterations(n::Integer)
    # About 4x the measured 206-246 band. A run that reaches this has
    # stagnated and belongs on the LU path, not on more iterations.
    return _beat_env_int(BEAT_GMRES_MAX_ITERATIONS_ENV, min(Int(n), 1000))
end

_beat_gmres_restart() = _beat_env_int(BEAT_GMRES_RESTART_ENV, 0; allow_zero=true)

# --- Router -----------------------------------------------------------------

"""
    beat_solve_dense_system(matrix, rhs; method=..., preserve_matrix=true)

Solve `matrix * X = rhs` for every column of `rhs`, choosing dense LU or
diagonally preconditioned GMRES by cost model, and falling back from GMRES to
the LU if any drive fails to converge.

Returns `(solution, report)`. The report carries the plan, the method actually
used, per-drive iteration counts and relative residuals, and `fell_back` when
GMRES was chosen and did not deliver.

`preserve_matrix` is true because the fused Metal path hands over a shared
device buffer the caller still owns; `lu!` would overwrite it. GMRES never
writes to the matrix, so choosing GMRES also avoids the N^2 complex copy the
LU path needs -- 3.3 GB at 20,422 dofs.
"""
function beat_solve_dense_system(matrix::AbstractMatrix{Complex{T}},
                                 rhs::AbstractVecOrMat{Complex{T}};
                                 method::Symbol=beat_dense_solve_method(),
                                 preserve_matrix::Bool=true) where {T<:AbstractFloat}
    rhs_matrix = rhs isa AbstractMatrix ? rhs : reshape(rhs, :, 1)
    n = size(matrix, 1)
    drive_count = size(rhs_matrix, 2)
    plan = beat_dense_solve_plan(n, drive_count; method=method)

    if plan.method === :gmres
        solution = similar(rhs_matrix, Complex{T}, n, drive_count)
        fill!(solution, zero(Complex{T}))
        inverse_diagonal = beat_diagonal_preconditioner(matrix)
        iterations = Int[]
        residuals = T[]
        converged = true
        # A model-chosen GMRES gets one LU's worth of matvecs across every
        # drive; an explicitly requested one is left alone. `ceiling` is the
        # hard stall guard and still applies.
        ceiling = _beat_gmres_max_iterations(n)
        remaining = plan.reason === :model ?
                    beat_gmres_iteration_budget(n, drive_count) : typemax(Int)
        gmres_elapsed = @elapsed for drive in 1:drive_count
            column = view(solution, :, drive)
            result = beat_gmres!(column, matrix, Vector{Complex{T}}(view(rhs_matrix, :, drive));
                                 preconditioner=inverse_diagonal,
                                 max_iterations=min(ceiling, max(remaining, 1)))
            push!(iterations, result.iterations)
            push!(residuals, result.relative_residual)
            remaining -= result.iterations
            if !result.converged
                # The fallback re-solves every column against one factorization,
                # so any Krylov work on the remaining drives would be discarded.
                converged = false
                break
            end
        end
        if converged
            return solution, (
                plan=plan,
                method=:gmres,
                fell_back=false,
                iterations=iterations,
                relative_residuals=residuals,
                seconds=gmres_elapsed,
            )
        end
        # Non-convergence is reported, never raised. The dense LU always
        # answers on this operator, and a slower correct solve beats a crash.
        @warn "BEAT GMRES did not converge; falling back to the dense LU." dofs = n drives = drive_count iterations = iterations relative_residuals = residuals
        lu_solution, lu_elapsed = _beat_dense_lu_solve(matrix, rhs_matrix, preserve_matrix)
        return lu_solution, (
            plan=plan,
            method=:lu,
            fell_back=true,
            iterations=iterations,
            relative_residuals=residuals,
            seconds=gmres_elapsed + lu_elapsed,
        )
    end

    solution, elapsed = _beat_dense_lu_solve(matrix, rhs_matrix, preserve_matrix)
    return solution, (
        plan=plan,
        method=:lu,
        fell_back=false,
        iterations=Int[],
        relative_residuals=T[],
        seconds=elapsed,
    )
end

function _beat_dense_lu_solve(matrix, rhs_matrix, preserve_matrix::Bool)
    elapsed = @elapsed begin
        factorization = preserve_matrix ? lu!(copy(matrix)) : lu!(matrix)
        solution = factorization \ rhs_matrix
    end
    return solution, elapsed
end

"""
    describe_dense_solve(report)

One line for the run's diagnostics: what was solved, how, and -- when GMRES
ran -- how hard it had to work. A fallback says so explicitly, because a
Krylov path that silently degrades to a direct solve is a performance
regression nobody attributes correctly six weeks later.
"""
function describe_dense_solve(report)
    plan = report.plan
    selection = plan.reason === :override ? "forced" : "cost model"
    if report.method === :gmres
        iterations = isempty(report.iterations) ? 0 : maximum(report.iterations)
        residual = isempty(report.relative_residuals) ? 0.0 : maximum(report.relative_residuals)
        return "Julia GMRES, diagonal preconditioner ($selection): $(plan.dofs) dofs, " *
            "$(plan.drives) drive(s), up to $iterations iterations, " *
            "relative residual $(round(residual; sigdigits=3))"
    end
    if report.fell_back
        iterations = isempty(report.iterations) ? 0 : maximum(report.iterations)
        residual = isempty(report.relative_residuals) ? 0.0 : maximum(report.relative_residuals)
        return "Julia direct dense solve after GMRES failed to converge " *
            "($iterations iterations, relative residual $(round(residual; sigdigits=3))): " *
            "$(plan.dofs) dofs, $(plan.drives) drive(s)"
    end
    return "Julia direct dense solve ($selection): $(plan.dofs) dofs, $(plan.drives) drive(s)"
end
