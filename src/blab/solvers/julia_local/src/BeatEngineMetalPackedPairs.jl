# Packed pair helpers for fused exterior Burton-Miller assembly and fast field evaluation.
#
# The pair maths is the same arithmetic as `_metal_regular_pair_blocks_inbounds`,
# but the rule points, weights and basis values are compile-time constants (a
# Val tuple), and each quadrature point, normal and curl row is one float4 load:
#   points4[(face - 1) * R + q] = (x, y, z, 0)
#   normals4[face]              = (nx, ny, nz, 0)
#   curls4[(face - 1) * 3 + a]  = curl of basis a, (x, y, z, 0)
# Packing copies the geometry; fast arithmetic and fused accumulation can change
# Float32 rounding. Only the helpers needed by the exterior and field kernels live here.

const _MetalFloat4 = NTuple{4,VecElement{Float32}}

@inline _metal_float4(a, b, c, d) = (VecElement(a), VecElement(b), VecElement(c), VecElement(d))

struct MetalPackedPairTables
    points4
    normals4
    curls4
    rule::Tuple   # (xi_1..xi_R, eta_1..eta_R, w_1..w_R), Float32
end

# Protect lazy read-only geometry tables; callers own all writable scratch.
# Cache release requires all assemblies/field calls to have completed.
const _metal_packed_cache_lock = ReentrantLock()

const _metal_packed_pair_tables = WeakKeyDict{Any,MetalPackedPairTables}()

function _metal_packed_pair_tables_for(cache::MetalRegularAssemblyCache)
    # Keyed by the cache's mutable gather_tables Ref, so it lives as long as the cache.
    lock(_metal_packed_cache_lock) do
        get!(_metal_packed_pair_tables, cache.gather_tables) do
            R = cache.rule_count
            F = cache.face_count
            erp = Array(cache.element_rule_points)   # flat: face + F*(q-1) + F*R*(c-1)
            nrm = Array(cache.normals)
            crl = Array(cache.curls)
            points4 = Vector{_MetalFloat4}(undef, F * R)
            normals4 = Vector{_MetalFloat4}(undef, F)
            curls4 = Vector{_MetalFloat4}(undef, F * 3)
            for face in 1:F
                for q in 1:R
                    p = face + F * (q - 1)
                    points4[(face - 1) * R + q] = _metal_float4(erp[p], erp[p + F * R], erp[p + 2 * F * R], 0.0f0)
                end
                normals4[face] = _metal_float4(nrm[face], nrm[face + F], nrm[face + 2F], 0.0f0)
                for a in 1:3
                    b = face + 3 * (a - 1) * F
                    curls4[(face - 1) * 3 + a] = _metal_float4(crl[b], crl[b + F], crl[b + 2F], 0.0f0)
                end
            end
            rp = Array(cache.rule_points)
            rw = Array(cache.rule_weights)
            rule = (Float32.(rp[1:R])..., Float32.(rp[R+1:2R])..., Float32.(rw[1:R])...)
            MetalPackedPairTables(MtlArray(points4), MtlArray(normals4), MtlArray(curls4), rule)
        end
    end
end

function _release_metal_packed_pair_tables!(cache::MetalRegularAssemblyCache)
    tables = lock(() -> pop!(_metal_packed_pair_tables, cache.gather_tables, nothing), _metal_packed_cache_lock)
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.points4)
    Metal.unsafe_free!(tables.normals4)
    Metal.unsafe_free!(tables.curls4)
    return nothing
end

@inline _metal_rule_xi(::Val{RC}, ::Val{R}, q) where {RC,R} = RC[q]
@inline _metal_rule_eta(::Val{RC}, ::Val{R}, q) where {RC,R} = RC[R + q]
@inline _metal_rule_w(::Val{RC}, ::Val{R}, q) where {RC,R} = RC[2R + q]

@inline _metal_packed_trial_fold(acc, context, ::Val{0}, rc, rv) = acc
@inline function _metal_packed_trial_fold(acc, context, ::Val{N}, rc, rv) where {N}
    acc = _metal_packed_trial_fold(acc, context, Val(N - 1), rc, rv)
    return _metal_packed_trial_term(acc, context, Val(N), rc, rv)
end

@inline function _metal_packed_trial_term(acc, context, ::Val{Q}, rc, rv::Val{R}) where {Q,R}
    s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = acc
    x, y, z, test_weight_scale, k, inv_four_pi,
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, trial_index = context
    xi = _metal_rule_xi(rc, rv, Q)
    eta = _metal_rule_eta(rc, rv, Q)
    rb1 = one(k) - xi - eta
    trial_weight = _metal_rule_w(rc, rv, Q)
    @inbounds p = points4[(trial_index - Int32(1)) * Int32(R) + Int32(Q)]
    sx = p[1].value
    sy = p[2].value
    sz = p[3].value
    Base.@fastmath begin
    dx = sx * trial_signs[1] - x
    dy = sy * trial_signs[2] - y
    dz = sz * trial_signs[3] - z
    radius2 = dx * dx + dy * dy + dz * dz
    if radius2 > zero(k)
        rb = SVector(rb1, xi, eta)
        inv_radius = _metal_fast_rsqrt(radius2)
        radius = radius2 * inv_radius
        phase = k * radius
        green_scale = inv_radius * inv_four_pi * (test_weight_scale * trial_weight)
        green_re = _metal_fast_cos(phase) * green_scale
        green_im = _metal_fast_sin(phase) * green_scale
        grad_re = -green_re * inv_radius - green_im * k
        grad_im = green_re * k - green_im * inv_radius
        test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
        trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
        s_re += green_re
        s_im += green_im
        a_re += grad_re * test_dot
        a_im += grad_im * test_dot
        d_re += rb * (grad_re * trial_dot)
        d_im += rb * (grad_im * trial_dot)
        h_re += rb * green_re
        h_im += rb * green_im
    end
    end
    return (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im)
end
