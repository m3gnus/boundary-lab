# Kernels of the fused Burton-Miller exterior assembly (BeatEngineMetalBurtonMiller.jl).
#
# Regular pairs: the packed pair maths of BeatEngineMetalPackedPairs.jl (float4 points, normals and
# curls, rule constants as a Val tuple, the trial loop unrolled), with the Burton-Miller combination
# formed per test point. Symmetry images add into the same pair blocks (`ACC`), so the gathers run
# once per chunk.
#
# Singular pairs: the Sauter-Schwab blocks with packed rule points and vertices, grouped by rule
# length. Every frequency retains the full original singular quadrature.

# The test-point loop stays a runtime loop (rule values from the device arrays); only the trial
# fold is unrolled. Unrolling both loops made the kernel slower.
@inline function _metal_fused_packed_test_loop(acc, context, rule_points, rule_weights, rc, rv::Val{R}) where {R}
    lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im = acc
    k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index, inverse_k, curl_scale = context
    T = typeof(k)
    z3 = zero(SVector{3,T})
    test_q = Int32(1)
    while test_q <= Int32(R)
        @inbounds test_xi = rule_points[test_q]
        @inbounds test_eta = rule_points[test_q + Int32(R)]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        @inbounds p = points4[(test_index - Int32(1)) * Int32(R) + test_q]
        @inbounds test_weight = rule_weights[test_q]
        trial_context = (p[1].value, p[2].value, p[3].value, test_weight * jac_scale, k, inv_four_pi,
            test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs, points4, trial_index)
        s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_packed_trial_fold(
            (zero(k), zero(k), zero(k), zero(k), z3, z3, z3, z3), trial_context, Val(R), rc, rv)
        rhs_re += test_basis * (-s_re + inverse_k * a_im)
        rhs_im += test_basis * (-s_im - inverse_k * a_re)
        g_total_re += s_re
        g_total_im += s_im
        u_re = -d_re + curl_scale * h_im
        u_im = -d_im - curl_scale * h_re
        lhs_re += SVector(
            tb1 * u_re[1], tb2 * u_re[1], tb3 * u_re[1],
            tb1 * u_re[2], tb2 * u_re[2], tb3 * u_re[2],
            tb1 * u_re[3], tb2 * u_re[3], tb3 * u_re[3],
        )
        lhs_im += SVector(
            tb1 * u_im[1], tb2 * u_im[1], tb3 * u_im[1],
            tb1 * u_im[2], tb2 * u_im[2], tb3 * u_im[2],
            tb1 * u_im[3], tb2 * u_im[3], tb3 * u_im[3],
        )
        test_q += Int32(1)
    end
    return (lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im)
end

function _metal_fused_pair_blocks_kernel!(
    blocks, points4, normals4, areas, curls4, faces, elements, rule_points, rule_weights,
    element_count::Int32, chunk_start::Int32, chunk_count::Int32, pair_stride::Int32,
    k, inverse_k, face_count::Int32, rc, rv::Val{R},
    pair_offsets, singular_trial_indices, skip_mode,
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    ::Val{ACC},
) where {R,ACC}
    position = thread_position_in_grid_2d()
    test_position = Int32(position.x)
    trial_local = Int32(position.y)
    (test_position > element_count || trial_local > chunk_count) && return nothing
    @inbounds test_index = Int32(elements[test_position])
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    base = test_position + element_count * (trial_local - Int32(1))
    if _metal_pair_is_skipped(faces, face_count, test_index, trial_index, pair_offsets, singular_trial_indices, skip_mode)
        ACC && return nothing   # a later transform adds nothing to a skipped pair
        component = Int32(0)
        while component < Int32(_METAL_FUSED_COMPONENTS)
            @inbounds blocks[base + component * pair_stride] = zero(eltype(blocks))
            component += Int32(1)
        end
        return nothing
    end
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    @inbounds tn = normals4[test_index]
    @inbounds rn = normals4[trial_index]
    test_nx = tn[1].value
    test_ny = tn[2].value
    test_nz = tn[3].value
    trial_nx = trial_sign_x * rn[1].value
    trial_ny = trial_sign_y * rn[2].value
    trial_nz = trial_sign_z * rn[3].value
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    @inbounds jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)
    curl_scale = inverse_k * k * k * normal_product
    context = (k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index, inverse_k, curl_scale)
    acc = (zero(SVector{9,T}), zero(SVector{9,T}), zero(SVector{3,T}), zero(SVector{3,T}), zero(k), zero(k))
    lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im =
        _metal_fused_packed_test_loop(acc, context, rule_points, rule_weights, rc, rv)
    @inbounds t1 = curls4[(test_index - Int32(1)) * Int32(3) + Int32(1)]
    @inbounds t2 = curls4[(test_index - Int32(1)) * Int32(3) + Int32(2)]
    @inbounds t3 = curls4[(test_index - Int32(1)) * Int32(3) + Int32(3)]
    @inbounds q1 = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(1)]
    @inbounds q2 = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(2)]
    @inbounds q3 = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(3)]
    t11, t12, t13 = t1[1].value, t1[2].value, t1[3].value
    t21, t22, t23 = t2[1].value, t2[2].value, t2[3].value
    t31, t32, t33 = t3[1].value, t3[2].value, t3[3].value
    r11 = trial_curl_sign_x * q1[1].value
    r12 = trial_curl_sign_y * q1[2].value
    r13 = trial_curl_sign_z * q1[3].value
    r21 = trial_curl_sign_x * q2[1].value
    r22 = trial_curl_sign_y * q2[2].value
    r23 = trial_curl_sign_z * q2[3].value
    r31 = trial_curl_sign_x * q3[1].value
    r32 = trial_curl_sign_y * q3[2].value
    r33 = trial_curl_sign_z * q3[3].value
    curl_products = SVector(
        t11 * r11 + t12 * r12 + t13 * r13,
        t21 * r11 + t22 * r12 + t23 * r13,
        t31 * r11 + t32 * r12 + t33 * r13,
        t11 * r21 + t12 * r22 + t13 * r23,
        t21 * r21 + t22 * r22 + t23 * r23,
        t31 * r21 + t32 * r22 + t33 * r23,
        t11 * r31 + t12 * r32 + t13 * r33,
        t21 * r31 + t22 * r32 + t23 * r33,
        t31 * r31 + t32 * r32 + t33 * r33,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    if ACC
        _metal_add_block!(blocks, base, pair_stride, Int32(0), lhs_re)
        _metal_add_block!(blocks, base, pair_stride, Int32(9), lhs_im)
        _metal_add_block!(blocks, base, pair_stride, Int32(18), rhs_re)
        _metal_add_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    else
        _metal_store_block!(blocks, base, pair_stride, Int32(0), lhs_re)
        _metal_store_block!(blocks, base, pair_stride, Int32(9), lhs_im)
        _metal_store_block!(blocks, base, pair_stride, Int32(18), rhs_re)
        _metal_store_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    end
    return nothing
end

@inline function _metal_add_block!(blocks, base::Int32, stride::Int32, offset::Int32, values::SVector{N,T}) where {N,T}
    i = 1
    while i <= N
        @inbounds blocks[base + (offset + Int32(i - 1)) * stride] += values[i]
        i += 1
    end
    return nothing
end

# Fused singular blocks with a load diet. The maths and the summation order are those of
# the original fused Duffy kernel, but
#   - the pairs are grouped once per singular cache by rule length (coincident / edge / vertex rules
#     each have one point count), one launch per group with the point and part counts as Vals;
#   - each rule point is one float4 (test xi, test eta, trial xi, trial eta);
#   - face vertices, normals and curl rows are float4 rows (3 / 1 / 3 per face).
#   - both faces' vertices are loaded once per thread.
# Outputs keep the (pair, part) layout the gathers read.

struct MetalFusedSingularTables
    groups::Vector{Tuple{Int,Any}}   # (rule point count, device Int32 pair positions)
    rule_points4
    vertices4
end

const _metal_fused_singular_tables = IdDict{Any,MetalFusedSingularTables}()

function _metal_fused_singular_tables_for(regular_cache, singular_cache)
    # Identity-keyed by the singular cache's rule_weights array; released with the cache.
    lock(_metal_packed_cache_lock) do
        get!(_metal_fused_singular_tables, singular_cache.rule_weights) do
            F = regular_cache.face_count
            fv = Array(regular_cache.face_vertices)
            vertices4 = Vector{_MetalFloat4}(undef, 3F)
            for face in 1:F, v in 1:3
                b = face + 3 * (v - 1) * F
                vertices4[(face - 1) * 3 + v] = _metal_float4(Float32(fv[b]), Float32(fv[b + F]), Float32(fv[b + 2F]), 0.0f0)
            end
            tp = vec(Array(singular_cache.rule_test_points))
            rp = vec(Array(singular_cache.rule_trial_points))
            n = length(singular_cache.rule_weights)
            points4 = [_metal_float4(Float32(tp[q]), Float32(tp[q + n]), Float32(rp[q]), Float32(rp[q + n])) for q in 1:n]
            offsets = Array(singular_cache.rule_offsets)
            by_length = Dict{Int,Vector{Int32}}()
            for (position, rule) in enumerate(Array(singular_cache.rule_indices))
                push!(get!(by_length, Int(offsets[rule + 1] - offsets[rule]), Int32[]), Int32(position))
            end
            groups = Tuple{Int,Any}[(len, MtlArray(by_length[len])) for len in sort!(collect(keys(by_length)))]
            MetalFusedSingularTables(groups, MtlArray(points4), MtlArray(vertices4))
        end
    end
end

@inline function _metal_face_point4(v1, v2, v3, basis1, basis2, basis3)
    x = basis1 * v1[1].value + basis2 * v2[1].value + basis3 * v3[1].value
    y = basis1 * v1[2].value + basis2 * v2[2].value + basis3 * v3[2].value
    z = basis1 * v1[3].value + basis2 * v2[3].value + basis3 * v3[3].value
    return x, y, z
end

@inline function _metal_pair_curl_products4(curls4, test_index, trial_index, csx, csy, csz)
    @inbounds begin
        ta = curls4[(test_index - Int32(1)) * Int32(3) + Int32(1)]
        tb = curls4[(test_index - Int32(1)) * Int32(3) + Int32(2)]
        tc = curls4[(test_index - Int32(1)) * Int32(3) + Int32(3)]
        ra = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(1)]
        rb = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(2)]
        rc = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(3)]
    end
    t11 = ta[1].value; t12 = ta[2].value; t13 = ta[3].value
    t21 = tb[1].value; t22 = tb[2].value; t23 = tb[3].value
    t31 = tc[1].value; t32 = tc[2].value; t33 = tc[3].value
    r11 = csx * ra[1].value; r12 = csy * ra[2].value; r13 = csz * ra[3].value
    r21 = csx * rb[1].value; r22 = csy * rb[2].value; r23 = csz * rb[3].value
    r31 = csx * rc[1].value; r32 = csy * rc[2].value; r33 = csz * rc[3].value
    return SVector(
        t11 * r11 + t12 * r12 + t13 * r13,
        t21 * r11 + t22 * r12 + t23 * r13,
        t31 * r11 + t32 * r12 + t33 * r13,
        t11 * r21 + t12 * r22 + t13 * r23,
        t21 * r21 + t22 * r22 + t23 * r23,
        t31 * r21 + t32 * r22 + t33 * r23,
        t11 * r31 + t12 * r32 + t13 * r33,
        t21 * r31 + t22 * r32 + t23 * r33,
        t31 * r31 + t32 * r32 + t33 * r33,
    )
end

function _metal_fused_singular_packed_kernel!(
    lhs_values, rhs_values, positions,
    test_indices, trial_indices, rule_indices, jac_scales, normal_products, rule_offsets,
    points4, rule_weights, vertices4, normals4, curls4,
    k::Float32, inverse_k::Float32, group_count::Int32, pair_count::Int32,
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    ::Val{N}, ::Val{P},
) where {N,P}
    thread = Int32(thread_position_in_grid_1d())
    thread > group_count * Int32(P) && return nothing
    group_position = (thread - Int32(1)) % group_count + Int32(1)
    part = (thread - Int32(1)) ÷ group_count + Int32(1)
    T = Float32
    @inbounds begin
        pair_position = positions[group_position]
        test_index = Int32(test_indices[pair_position])
        trial_index = Int32(trial_indices[pair_position])
        rule_index = Int32(rule_indices[pair_position])
        q_first = Int32(rule_offsets[rule_index])
        q_last = q_first + Int32(N) - Int32(1)
        per_part = Int32(cld(N, P))
        q = q_first + (part - Int32(1)) * per_part
        q_stop = min(q + per_part - Int32(1), q_last)
        jac_scale = jac_scales[pair_position]
        normal_product = normal_products[pair_position]
        tn = normals4[test_index]
        rn = normals4[trial_index]
        test_nx = tn[1].value
        test_ny = tn[2].value
        test_nz = tn[3].value
        trial_nx = trial_sign_x * rn[1].value
        trial_ny = trial_sign_y * rn[2].value
        trial_nz = trial_sign_z * rn[3].value
        test_row = (test_index - Int32(1)) * Int32(3)
        trial_row = (trial_index - Int32(1)) * Int32(3)
        tv1 = vertices4[test_row + Int32(1)]; tv2 = vertices4[test_row + Int32(2)]; tv3 = vertices4[test_row + Int32(3)]
        sv1 = vertices4[trial_row + Int32(1)]; sv2 = vertices4[trial_row + Int32(2)]; sv3 = vertices4[trial_row + Int32(3)]
    end
    inv_four_pi = T(0.07957747154594767)
    curl_scale = inverse_k * k * k * normal_product
    lhs_re = zero(SVector{9,T}); lhs_im = zero(SVector{9,T})
    rhs_re = zero(SVector{3,T}); rhs_im = zero(SVector{3,T})
    g_total_re = zero(T); g_total_im = zero(T)
    while q <= q_stop
        @inbounds begin
            rule_point = points4[q]
            weight = rule_weights[q] * jac_scale
        end
        test_xi = rule_point[1].value
        test_eta = rule_point[2].value
        trial_xi = rule_point[3].value
        trial_eta = rule_point[4].value
        tb1 = one(k) - test_xi - test_eta
        rb1 = one(k) - trial_xi - trial_eta
        x, y, z = _metal_face_point4(tv1, tv2, tv3, tb1, test_xi, test_eta)
        sx, sy, sz = _metal_face_point4(sv1, sv2, sv3, rb1, trial_xi, trial_eta)
        Base.@fastmath begin
            dx = sx * trial_sign_x - x
            dy = sy * trial_sign_y - y
            dz = sz * trial_sign_z - z
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 > zero(k)
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
                phase = k * radius
                green_scale = inv_radius * inv_four_pi * weight
                green_re = _metal_fast_cos(phase) * green_scale
                green_im = _metal_fast_sin(phase) * green_scale
                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                tb = SVector(tb1, test_xi, test_eta)
                outer = SVector(
                    tb1 * rb1, test_xi * rb1, test_eta * rb1,
                    tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                    tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
                )
                rhs_re += tb * (-green_re + inverse_k * (grad_im * test_dot))
                rhs_im += tb * (-green_im - inverse_k * (grad_re * test_dot))
                u_re = -(grad_re * trial_dot) + curl_scale * green_im
                u_im = -(grad_im * trial_dot) - curl_scale * green_re
                lhs_re += outer * u_re
                lhs_im += outer * u_im
                g_total_re += green_re
                g_total_im += green_im
            end
        end
        q += Int32(1)
    end
    curl_products = _metal_pair_curl_products4(
        curls4, test_index, trial_index, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    linear_index = pair_position + (part - Int32(1)) * pair_count
    value_stride = pair_count * Int32(P)
    @inbounds begin
        i = 1
        while i <= 3
            rhs_values[linear_index + Int32(i - 1) * value_stride] = Complex(rhs_re[i], rhs_im[i])
            i += 1
        end
        i = 1
        while i <= 9
            lhs_values[linear_index + Int32(i - 1) * value_stride] = Complex(lhs_re[i], lhs_im[i])
            i += 1
        end
    end
    return nothing
end

# Release only after every assembly using this cache has completed.
function _release_metal_fused_singular_tables!(singular_cache)
    tables = lock(() -> pop!(_metal_fused_singular_tables, singular_cache.rule_weights, nothing), _metal_packed_cache_lock)
    if tables !== nothing
        foreach(group -> Metal.unsafe_free!(group[2]), tables.groups)
        Metal.unsafe_free!(tables.rule_points4)
        Metal.unsafe_free!(tables.vertices4)
    end
    return nothing
end
