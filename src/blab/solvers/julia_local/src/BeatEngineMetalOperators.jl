# Operator storage, symmetry row weights, the Burton-Miller system, and the
# dense solve for Metal-assembled operators.
#
# Metal.jl has no GPU LU, so the dense factorization runs on the CPU. Apple
# Silicon is a unified-memory device, so the operators are not copied at all:
# they are allocated in shared storage and wrapped as host `Array`s in place.
# The CPU Burton-Miller system code is reused unchanged, which also gives one
# factorization per frequency shared across every channel drive, the property
# the CPU backend has and the CUDA path lacks.

const _METAL_OPERATOR_KEYS = (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)

"""
    release_operator_storage!(operators)

Free the four dense Burton-Miller operator buffers.

Release exactly once, through whichever tuple you still hold: a device tuple
frees its own arrays, and a host tuple from `metal_host_operators` frees the
shared-storage buffers its views alias (`metal_backing`). Releasing the device
tuple while host views over it are still in use leaves those views dangling.
"""
function release_operator_storage!(operators::NamedTuple)
    backing = get(operators, :metal_backing, nothing)
    if backing !== nothing
        for key in _METAL_OPERATOR_KEYS
            Metal.unsafe_free!(getfield(backing, key))
        end
        return nothing
    end
    get(operators, :on_gpu, false) || return nothing
    get(operators, :gpu_backend, nothing) == :metal || return nothing
    for key in _METAL_OPERATOR_KEYS
        Metal.unsafe_free!(getfield(operators, key))
    end
    return nothing
end

function _apply_metal_operator_p1_row_weights!(operators, mesh::BoundaryMesh{T}, symmetry_mode) where {T<:AbstractFloat}
    normalized_symmetry_mode(symmetry_mode) == :off && return nothing
    d_weights = MtlArray(Complex{T}.(p1_symmetry_orbit_weights(mesh, symmetry_mode)))
    operators.single_layer .*= reshape(d_weights, :, 1)
    operators.double_layer .*= reshape(d_weights, :, 1)
    operators.adjoint_double_layer .*= reshape(d_weights, :, 1)
    operators.hypersingular .*= reshape(d_weights, :, 1)
    Metal.synchronize()
    Metal.unsafe_free!(d_weights)
    return nothing
end

"""
    metal_host_operators(operators)

Present Metal-resident operators to the host as a NamedTuple with the same
keys and `on_gpu=false`, so every CPU solve routine accepts it.

Shared-storage buffers are wrapped in place, so this costs nothing and the
returned arrays alias device memory: the returned tuple carries the device
arrays under `metal_backing` and takes ownership of them, so
`release_operator_storage!` on *it* is what frees them. Release exactly once,
through whichever tuple you still hold: freeing the device tuple while the host
views are alive leaves them dangling. Private-storage buffers are copied as
before, and then the returned tuple owns nothing and the device tuple is still
the caller's to free.
"""
function metal_host_operators(operators::NamedTuple)
    get(operators, :gpu_backend, nothing) == :metal || error("metal_host_operators requires Metal operators.")
    Metal.synchronize()
    shared = all(Metal.is_shared(getfield(operators, key)) for key in _METAL_OPERATOR_KEYS)
    host = if shared
        (
            single_layer=unsafe_wrap(Array, operators.single_layer),
            double_layer=unsafe_wrap(Array, operators.double_layer),
            adjoint_double_layer=unsafe_wrap(Array, operators.adjoint_double_layer),
            hypersingular=unsafe_wrap(Array, operators.hypersingular),
        )
    else
        (
            single_layer=Array(operators.single_layer),
            double_layer=Array(operators.double_layer),
            adjoint_double_layer=Array(operators.adjoint_double_layer),
            hypersingular=Array(operators.hypersingular),
        )
    end
    backing = shared ? NamedTuple{_METAL_OPERATOR_KEYS}(
        map(key -> getfield(operators, key), _METAL_OPERATOR_KEYS),
    ) : nothing
    extras = Base.structdiff(operators, NamedTuple{(:single_layer, :double_layer, :adjoint_double_layer, :hypersingular, :on_gpu, :metal_backing)})
    return merge(extras, host, (on_gpu=false, host_copy_of=:metal, metal_backing=backing))
end

function build_metal_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, ::Type{T}) where {T<:AbstractFloat}
    _require_metal!()
    # The dense solve is host-side, so the identity blocks stay on the host.
    return MetalBurtonMillerIdentityCache(
        Complex{T}.(identity_p1_p1),
        Complex{T}.(identity_p1_dp0),
    )
end

release_metal_burton_miller_identity_cache!(::MetalBurtonMillerIdentityCache) = nothing

struct MetalSparseScatterCache{R,C,V}
    rows::R
    columns::C
    values::V
end

function build_metal_sparse_scatter_cache(matrix::SparseMatrixCSC)
    _require_metal!()
    rows, columns, values = findnz(matrix)
    return MetalSparseScatterCache(
        MtlArray(Int32.(rows)),
        MtlArray(Int32.(columns)),
        MtlArray(values),
    )
end

function _metal_sparse_scatter_kernel!(
    destination,
    rows,
    columns,
    values,
    row_offset,
    column_offset,
    alpha,
    add,
)
    index = _metal_global_linear_index()
    if index <= length(values)
        row = Int(rows[index]) + row_offset
        column = Int(columns[index]) + column_offset
        value = alpha * values[index]
        if add
            destination[row, column] += value
        else
            destination[row, column] = value
        end
    end
    return nothing
end

function scatter_metal_sparse_to_dense!(
    destination,
    cache::MetalSparseScatterCache;
    row_offset::Integer=0,
    column_offset::Integer=0,
    alpha=one(eltype(destination)),
    add::Bool=false,
)
    isempty(cache.values) && return destination
    _metal_launch(
        _metal_sparse_scatter_kernel!,
        length(cache.values),
        destination,
        cache.rows,
        cache.columns,
        cache.values,
        Int(row_offset),
        Int(column_offset),
        convert(eltype(destination), alpha),
        add,
    )
    return destination
end

function release_metal_sparse_scatter_cache!(cache::MetalSparseScatterCache)
    Metal.unsafe_free!(cache.rows)
    Metal.unsafe_free!(cache.columns)
    Metal.unsafe_free!(cache.values)
    return nothing
end

"""
    metal_dense_lu!(matrix)

Factor a Metal-resident (or host) dense matrix on the CPU. The device copy is
released; the returned factorization is a host LAPACK object.
"""
function metal_dense_lu!(matrix)
    if matrix isa MtlArray
        Metal.synchronize()
        host = Array(matrix)
        Metal.unsafe_free!(matrix)
        return lu!(host)
    end
    return lu!(matrix)
end

function solve_metal_dense_factorization(factorization, rhs)
    host_rhs = rhs isa MtlArray ? Array(rhs) : rhs
    return factorization \ host_rhs
end

function solve_burton_miller_neumann(
    operators,
    identity_cache::MetalBurtonMillerIdentityCache,
    q_neumann,
    k::T;
    coupling_cap::Real=zero(T),
) where {T<:AbstractFloat}
    get(operators, :on_gpu, false) || error("Cached Metal solve requires GPU-resident operators.")
    get(operators, :gpu_backend, nothing) == :metal || error("Cached Metal solve requires Metal operators.")
    _require_metal!()
    host_operators = metal_host_operators(operators)
    return solve_burton_miller_neumann_cpu(
        host_operators,
        identity_cache.identity_p1_p1,
        identity_cache.identity_p1_dp0,
        q_neumann,
        k;
        coupling_cap=coupling_cap,
    )
end
