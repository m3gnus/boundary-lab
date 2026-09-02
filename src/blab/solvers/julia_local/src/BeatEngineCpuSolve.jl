const BEAT_CPU_BLAS_THREADS_ENV = "BLAB_BEAT_CPU_BLAS_THREADS"

function beat_cpu_blas_thread_count(
    p1_dofs::Integer;
    available_threads::Integer=Threads.nthreads(),
    override::AbstractString=get(ENV, BEAT_CPU_BLAS_THREADS_ENV, "auto"),
)
    dof_count = max(1, Int(p1_dofs))
    thread_limit = max(1, Int(available_threads))
    override_text = lowercase(strip(override))
    if !isempty(override_text) && override_text != "auto"
        requested = try
            parse(Int, override_text)
        catch
            error("$BEAT_CPU_BLAS_THREADS_ENV must be 'auto' or a positive integer; got $(repr(override)).")
        end
        requested > 0 || error("$BEAT_CPU_BLAS_THREADS_ENV must be a positive integer; got $requested.")
        return min(requested, thread_limit)
    end

    if dof_count <= 768
        return 1
    elseif dof_count <= 2048
        return min(4, thread_limit)
    elseif dof_count <= 4096
        return min(8, thread_limit)
    end
    return thread_limit
end

function configure_beat_cpu_blas_threads!(
    p1_dofs::Integer;
    available_threads::Integer=Threads.nthreads(),
    override::AbstractString=get(ENV, BEAT_CPU_BLAS_THREADS_ENV, "auto"),
)
    thread_count = beat_cpu_blas_thread_count(
        p1_dofs;
        available_threads=available_threads,
        override=override,
    )
    BLAS.set_num_threads(thread_count)
    return BLAS.get_num_threads()
end

"""
    burton_miller_neumann_lhs(operators, identity_p1_p1, k; coupling_cap=0)

`0.5 M - D + eta H` as one fused broadcast, `eta = i |k| / max(k^2, cap)`.

The real identity block is broadcast directly against the complex operators
rather than promoted first: `Complex{T}.(identity_p1_p1)` materialised a full
N x N complex copy of a matrix that is 99.9% zeros -- 419 MB at 10,230 P1 dofs,
every frequency.
"""
function burton_miller_neumann_lhs(operators, identity_p1_p1, k::T; coupling_cap::Real=zero(T)) where {T<:AbstractFloat}
    coupling = burton_miller_coupling(k, coupling_cap)
    return Complex{T}(0.5) .* identity_p1_p1 .- operators.double_layer .+ coupling .* operators.hypersingular
end

"""
    burton_miller_neumann_rhs(operators, identity_p1_dp0, q_neumann, k; coupling_cap=0)

`(-S - eta (K' + 0.5 M_p1dp0)) q` without ever forming that operator.

The materialised form is N x 2N complex -- 1.67 GB at 10,230 P1 dofs -- built
once per frequency only to be multiplied by a vector. Three matrix-vector
products cost a fraction of writing it. This mirrors what the CUDA path already
does above 768 dofs (`_cuda_burton_miller_rhs`).
"""
function burton_miller_neumann_rhs(operators, identity_p1_dp0, q_neumann, k::T; coupling_cap::Real=zero(T)) where {T<:AbstractFloat}
    coupling = burton_miller_coupling(k, coupling_cap)
    q = Complex{T}.(q_neumann)
    rhs = Vector{Complex{T}}(undef, size(operators.single_layer, 1))
    mul!(rhs, operators.single_layer, q, -one(Complex{T}), zero(Complex{T}))
    mul!(rhs, operators.adjoint_double_layer, q, -coupling, one(Complex{T}))
    _add_scaled_matvec!(rhs, identity_p1_dp0, q, -T(0.5) * coupling)
    return rhs
end

# rhs .+= alpha .* (matrix * q).  The identity blocks are assembled real, and a
# real-matrix/complex-vector `mul!` falls off BLAS into a generic loop, so the
# real case is split into two BLAS gemv calls instead.
function _add_scaled_matvec!(rhs::Vector{Complex{T}}, matrix::AbstractMatrix{T},
                             q::Vector{Complex{T}}, alpha::Complex{T}) where {T<:AbstractFloat}
    real_part = matrix * real.(q)
    imag_part = matrix * imag.(q)
    @inbounds for index in eachindex(rhs)
        rhs[index] += alpha * Complex{T}(real_part[index], imag_part[index])
    end
    return rhs
end

function _add_scaled_matvec!(rhs::Vector{Complex{T}}, matrix::AbstractMatrix{Complex{T}},
                             q::Vector{Complex{T}}, alpha::Complex{T}) where {T<:AbstractFloat}
    return mul!(rhs, matrix, q, alpha, one(Complex{T}))
end

# Fallback for any other element type: correct, just not on a BLAS path.
function _add_scaled_matvec!(rhs::Vector{Complex{T}}, matrix::AbstractMatrix,
                             q::Vector{Complex{T}}, alpha::Complex{T}) where {T<:AbstractFloat}
    rhs .+= alpha .* (matrix * q)
    return rhs
end

function burton_miller_neumann_matrices(operators, identity_p1_p1, identity_p1_dp0, k::T; coupling_cap::Real=zero(T)) where {T<:AbstractFloat}
    coupling = burton_miller_coupling(k, coupling_cap)
    lhs = burton_miller_neumann_lhs(operators, identity_p1_p1, k; coupling_cap=coupling_cap)
    rhs_operator = -operators.single_layer .- coupling .* (operators.adjoint_double_layer .+ Complex{T}(0.5) .* identity_p1_dp0)
    return lhs, rhs_operator
end

function build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k::T; coupling_cap::Real=zero(T)) where {T<:AbstractFloat}
    return (
        factorization=lu!(burton_miller_neumann_lhs(operators, identity_p1_p1, k; coupling_cap=coupling_cap)),
        single_layer=operators.single_layer,
        adjoint_double_layer=operators.adjoint_double_layer,
        identity_p1_dp0=identity_p1_dp0,
        wavenumber=k,
        coupling_cap=T(coupling_cap),
    )
end

function solve_burton_miller_neumann_cpu_system(system, q_neumann, ::Type{T}) where {T<:AbstractFloat}
    rhs = burton_miller_neumann_rhs(system, system.identity_p1_dp0, q_neumann, system.wavenumber;
                                    coupling_cap=get(system, :coupling_cap, zero(system.wavenumber)))
    return Complex{T}.(system.factorization \ rhs)
end

function solve_burton_miller_neumann_cpu(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k::T; coupling_cap::Real=zero(T)) where {T<:AbstractFloat}
    system = build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k; coupling_cap=coupling_cap)
    return solve_burton_miller_neumann_cpu_system(system, q_neumann, T)
end
