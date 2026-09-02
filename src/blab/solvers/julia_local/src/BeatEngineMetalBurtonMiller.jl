# Fused Burton-Miller exterior assembly for Metal.
#
# The four-operator path assembles S, K', D and H separately and combines them
# on the host (`burton_miller_neumann_matrices`):
#
#   lhs    = 0.5 I_p1p1 - D + (i/k) H                     (N x N)
#   rhs_op = -S - (i/k) (K' + 0.5 I_p1dp0)                (N x 2N)
#
# eta = i/k is known at assembly time, so both combinations can be formed
# inside the pair kernel instead. This file does that: one N x N system matrix
# and one right-hand-side vector per drive, never the four operators.
#
# What that is worth, measured on the ATH ladder (see the head-to-head doc):
#
# * Memory: N^2 complex instead of 6N^2 (S and K' are N x 2N and are two thirds
#   of the total). That is the certain win and it is what sets the dense
#   ceiling.
# * Time: the gather stage halves its pair-buffer reads and its operator
#   write-backs, worth ~2.4x on that stage. The pair kernel itself gains
#   nothing: it is 82-84% arithmetic-bound, and the arithmetic that forms the
#   combination costs back exactly what halving the stores saves.
#
# The four-operator path stays: the coupled FEM/LEM solver needs the operators
# separately, and so would any Calderon preconditioner. This is an
# exterior-only fast path chosen at assembly time, not a replacement.
#
# Multi-drive: every channel's Neumann vector is folded in the same pass, so
# one assembly serves the whole channel set at a frequency exactly as one
# factorization does. `q_neumann` is dp0_count x drive_count.
#
# Component layout per pair (24 floats, against 48 for the four operators):
#   0-8   lhs re (3x3, column-major)   9-17  lhs im
#   18-20 rhs coefficient re (3 rows)  21-23 rhs coefficient im

const _METAL_FUSED_COMPONENTS = 24

struct MetalFusedGatherTables
    chunk_size::Int
    chunk_count::Int
    elements              # MtlArray{Int32}: assembly position -> global element
    element_positions     # MtlArray{Int32}: global element -> assembly position (0 if absent)
    chunk_node_offsets::Vector{Int}
    chunk_nodes           # MtlArray{Int32}: chunk-node position -> global P1 dof
    inc_offsets           # MtlArray{Int32}
    inc_packed            # MtlArray{Int32}: (trial local - 1) * 4 + local column
    blocks                # MtlArray{Float32}: 24 * element_count * chunk_size
end

function _metal_fused_chunk_size(element_count::Int)
    element_count <= 0 && return 1
    budget_mb = parse(Float64, get(ENV, "BLAB_METAL_GATHER_BUDGET_MB", "512"))
    per_column_bytes = element_count * _METAL_FUSED_COMPONENTS * sizeof(Float32)
    chunk = clamp(floor(Int, budget_mb * 1e6 / per_column_bytes), 1, element_count)
    override = strip(get(ENV, "BLAB_METAL_GATHER_CHUNK", ""))
    isempty(override) || (chunk = clamp(parse(Int, override), 1, element_count))
    while element_count * chunk * _METAL_FUSED_COMPONENTS >= typemax(Int32) && chunk > 1
        chunk = chunk ÷ 2
    end
    return chunk
end

function _metal_fused_gather_tables(cache::MetalRegularAssemblyCache)
    tables = cache.fused_gather_tables[]
    tables === nothing || return tables
    indices = cache.element_indices
    element_count = length(indices)
    chunk_size = _metal_fused_chunk_size(element_count)
    chunk_count = cld(element_count, chunk_size)
    p1_dofs = Array(cache.p1_dofs)
    element_positions = zeros(Int32, cache.face_count)
    for (position, element_index) in enumerate(indices)
        element_positions[element_index] = Int32(position)
    end
    chunk_node_offsets = Vector{Int}(undef, chunk_count + 1)
    chunk_node_offsets[1] = 1
    chunk_nodes = Int32[]
    inc_offsets = Int32[1]
    inc_packed = Int32[]
    for chunk in 1:chunk_count
        start = (chunk - 1) * chunk_size + 1
        stop = min(chunk * chunk_size, element_count)
        node_incidence = Dict{Int32,Vector{Int32}}()
        for position in start:stop
            element_index = indices[position]
            trial_local = position - start + 1
            for local_column in 1:3
                node = p1_dofs[element_index, local_column]
                push!(get!(node_incidence, node, Int32[]), Int32((trial_local - 1) * 4 + local_column))
            end
        end
        for node in sort!(collect(keys(node_incidence)))
            push!(chunk_nodes, node)
            append!(inc_packed, node_incidence[node])
            push!(inc_offsets, Int32(length(inc_packed) + 1))
        end
        chunk_node_offsets[chunk + 1] = length(chunk_nodes) + 1
    end
    blocks = MtlArray{Float32}(undef, _METAL_FUSED_COMPONENTS * element_count * chunk_size)
    tables = MetalFusedGatherTables(
        chunk_size,
        chunk_count,
        MtlArray(Int32.(indices)),
        MtlArray(element_positions),
        chunk_node_offsets,
        MtlArray(chunk_nodes),
        MtlArray(inc_offsets),
        MtlArray(inc_packed),
        blocks,
    )
    cache.fused_gather_tables[] = tables
    return tables
end

function _release_metal_fused_gather_tables!(cache::MetalRegularAssemblyCache)
    tables = cache.fused_gather_tables[]
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.elements)
    Metal.unsafe_free!(tables.element_positions)
    Metal.unsafe_free!(tables.chunk_nodes)
    Metal.unsafe_free!(tables.inc_offsets)
    Metal.unsafe_free!(tables.inc_packed)
    Metal.unsafe_free!(tables.blocks)
    cache.fused_gather_tables[] = nothing
    return nothing
end

# lhs contribution of a pair: -D + eta H with eta = i * s, so
#   re = -D_re - s H_im        im = -D_im + s H_re
# rhs coefficient of a pair: -S - eta K', so
#   re = -S_re + s K'_im       im = -S_im - s K'_re
# `s` is `burton_miller_coupling_scale(k, cap)`; it is 1/k uncapped.
# Identical algebra to `burton_miller_neumann_matrices`, formed per pair.
@inline function _metal_fused_pair_combination(
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im, coupling_scale,
)
    lhs_re = -dlp_re - coupling_scale * hyp_im
    lhs_im = -dlp_im + coupling_scale * hyp_re
    rhs_re = -slp_re + coupling_scale * adj_im
    rhs_im = -slp_im - coupling_scale * adj_re
    return lhs_re, lhs_im, rhs_re, rhs_im
end

# The pair's Burton-Miller contribution, combined inside the accumulation
# rather than after it.
#
# The per-quadrature-point-pair arithmetic cannot drop: D and H carry different
# geometric prefactors per entry (D is basis_product * grad * trial_dot, H is
# curl - k^2 * basis_product * n.n' against the Green's value), so both terms
# are evaluated whatever they are accumulated into. Same for S against K'.
#
# The rank-1 *outer products* are a different matter. `_metal_regular_pair_blocks`
# accumulates four 3-vectors per test point and then expands each into its own
# 3x3 block, four expansions per test point. The Burton-Miller combination is
# linear, so it can be applied to the 3-vectors *before* the expansion, leaving
# one 3x3 expansion instead of two and one 3x1 instead of two. That halves the
# outer-product work and the live 3x3 accumulators, 48 floats to 24.
#
# The hypersingular curl term is not inside the test loop at all: H is
# curl_products * G0 - k^2 (n.n') * (basis-weighted sums), and only the second
# half accumulates per test point. The first half is added once after the loop.
@inline function _metal_regular_pair_fused_blocks(
    face_vertices,
    normals,
    areas,
    curls,
    rule_points,
    rule_weights,
    element_rule_points,
    test_index::Int32,
    trial_index::Int32,
    face_count::Int32,
    k,
    coupling_scale,
    ::Val{R},
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {R}
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    @inbounds begin
    lhs_re = zero(SVector{9,T})
    lhs_im = zero(SVector{9,T})
    rhs_re = zero(SVector{3,T})
    rhs_im = zero(SVector{3,T})
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + Int32(2) * face_count]
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)
    # -eta * k^2 (n.n'), folded into the per-test-point combination.
    curl_scale = coupling_scale * k * k * normal_product

    g_total_re = zero(k)
    g_total_im = zero(k)
    test_q = Int32(1)
    while test_q <= Int32(R)
        test_xi = rule_points[test_q]
        test_eta = rule_points[test_q + Int32(R)]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        point_index = test_index + face_count * (test_q - Int32(1))
        x = element_rule_points[point_index]
        y = element_rule_points[point_index + face_count * Int32(R)]
        z = element_rule_points[point_index + face_count * Int32(2 * R)]
        test_weight = rule_weights[test_q]

        s_re = zero(k)
        s_im = zero(k)
        a_re = zero(k)
        a_im = zero(k)
        d_re = zero(SVector{3,T})
        d_im = zero(SVector{3,T})
        h_re = zero(SVector{3,T})
        h_im = zero(SVector{3,T})
        context = (x, y, z, test_weight * jac_scale, k, inv_four_pi,
            test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
            element_rule_points, rule_points, rule_weights, trial_index, face_count)
        s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_trial_fold(
            (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im),
            context, Val(R), Val(R),
        )
        # rhs coefficient: -S - (i/k) K'
        rhs_re += test_basis * (-s_re + coupling_scale * a_im)
        rhs_im += test_basis * (-s_im - coupling_scale * a_re)
        g_total_re += s_re
        g_total_im += s_im
        # lhs: -D + (i/k) H, less the curl term added after the loop.
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
    # (i/k) * curl_products * G0, computed after the loop so its nine values are
    # not live registers during it.
    curl_products = _metal_pair_curl_products(
        curls,
        test_index,
        trial_index,
        face_count,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    lhs_re -= curl_products * (coupling_scale * g_total_im)
    lhs_im += curl_products * (coupling_scale * g_total_re)
    return lhs_re, lhs_im, rhs_re, rhs_im
    end
end

function _metal_fused_pair_blocks_kernel!(
    blocks,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    rule_points,
    rule_weights,
    element_rule_points,
    elements,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    k,
    coupling_scale,
    face_count::Int32,
    ::Val{R},
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {R}
    position = thread_position_in_grid_2d()
    test_position = Int32(position.x)
    trial_local = Int32(position.y)
    (test_position > element_count || trial_local > chunk_count) && return nothing
    @inbounds test_index = Int32(elements[test_position])
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    base = test_position + element_count * (trial_local - Int32(1))
    if _metal_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    )
        component = Int32(0)
        while component < Int32(_METAL_FUSED_COMPONENTS)
            @inbounds blocks[base + component * pair_stride] = zero(eltype(blocks))
            component += Int32(1)
        end
        return nothing
    end
    lhs_re, lhs_im, rhs_re, rhs_im = _metal_regular_pair_fused_blocks(
        face_vertices,
        normals,
        areas,
        curls,
        rule_points,
        rule_weights,
        element_rule_points,
        test_index,
        trial_index,
        face_count,
        k,
        coupling_scale,
        Val(R),
        trial_sign_x,
        trial_sign_y,
        trial_sign_z,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    _metal_store_block!(blocks, base, pair_stride, Int32(0), lhs_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(9), lhs_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(18), rhs_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    return nothing
end

# One thread per (P1 row, P1 node touched by the chunk), exactly the ownership
# of `_metal_gather_dlp_hyp_kernel!` but reading two components instead of four
# and writing one matrix instead of two.
function _metal_fused_lhs_gather_kernel!(
    lhs,
    blocks,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    chunk_nodes,
    inc_offsets,
    inc_packed,
    node_start::Int32,
    node_count::Int32,
    element_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * node_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    node_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    node_position = node_start + node_local - Int32(1)
    @inbounds column = Int32(chunk_nodes[node_position])
    @inbounds chunk_first = Int32(inc_offsets[node_position])
    @inbounds chunk_stop = Int32(inc_offsets[node_position + Int32(1)]) - Int32(1)
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        chunk_position = chunk_first
        while chunk_position <= chunk_stop
            @inbounds packed = Int32(inc_packed[chunk_position])
            trial_local = (packed >> 2) + Int32(1)
            local_column = packed & Int32(3)
            pair = test_position + element_count * (trial_local - Int32(1))
            component = local_row + Int32(3) * (local_column - Int32(1))   # 1..9
            @inbounds value_re += blocks[pair + (component - Int32(1)) * pair_stride]
            @inbounds value_im += blocks[pair + (component + Int32(8)) * pair_stride]
            chunk_position += Int32(1)
        end
        incident_position += Int32(1)
    end
    @inbounds lhs[row + (column - Int32(1)) * p1_count] += Complex(value_re, value_im)
    return nothing
end

# One thread per (P1 row, trial element of the chunk, drive): the same
# ownership as `_metal_gather_slp_adjoint_kernel!`, but the result is a
# right-hand-side contribution rather than an operator column. Row `row` is
# touched by every trial element, so the sum over trial elements cannot happen
# here without a race; each (row, trial local) pair keeps its own partial and
# `_metal_fused_rhs_reduce_kernel!` sums them once at the end. The partial
# survives across chunks because (row, trial local) is the same owner in every
# chunk.
function _metal_fused_rhs_gather_kernel!(
    rhs_partial,
    blocks,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_dp0_dofs,
    q_neumann,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    chunk_size::Int32,
    pair_stride::Int32,
    p1_count::Int32,
    dp0_count::Int32,
    drive_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    column_base = element_count * (trial_local - Int32(1))
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        pair = test_position + column_base
        @inbounds value_re += blocks[pair + (local_row + Int32(17)) * pair_stride]
        @inbounds value_im += blocks[pair + (local_row + Int32(20)) * pair_stride]
        incident_position += Int32(1)
    end
    coefficient = Complex(value_re, value_im)
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    partial_index = row + (trial_local - Int32(1)) * p1_count
    drive = Int32(1)
    while drive <= drive_count
        @inbounds rhs_partial[partial_index + (drive - Int32(1)) * p1_count * chunk_size] +=
            coefficient * q_neumann[dp0_column + (drive - Int32(1)) * dp0_count]
        drive += Int32(1)
    end
    return nothing
end

function _metal_fused_rhs_reduce_kernel!(
    rhs,
    rhs_partial,
    p1_count::Int32,
    chunk_size::Int32,
    drive_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * drive_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    drive = (index - Int32(1)) ÷ p1_count + Int32(1)
    base = (drive - Int32(1)) * p1_count * chunk_size
    total = zero(eltype(rhs))
    column = Int32(1)
    while column <= chunk_size
        @inbounds total += rhs_partial[base + row + (column - Int32(1)) * p1_count]
        column += Int32(1)
    end
    @inbounds rhs[index] = total
    return nothing
end

function _launch_metal_fused_pair_kernels!(
    lhs,
    rhs_partial,
    q_neumann,
    cache::MetalRegularAssemblyCache,
    k,
    coupling_scale,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Fused Metal assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tables = _metal_fused_gather_tables(cache)
    chunk_size = tables.chunk_size
    pair_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    dp0_count = Int32(cache.dp0_dof_count)
    drive_count = Int32(size(q_neumann, 2))
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_fused_pair_blocks_kernel!(
            tables.blocks,
            cache.face_vertices,
            cache.normals,
            cache.areas,
            cache.faces,
            cache.curls,
            cache.rule_points,
            cache.rule_weights,
            cache.element_rule_points,
            tables.elements,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            k,
            coupling_scale,
            Int32(cache.face_count),
            Val(rule_count),
            pair_offsets,
            singular_trial_indices,
            skip_mode,
            trial_sign_x,
            trial_sign_y,
            trial_sign_z,
            trial_curl_sign_x,
            trial_curl_sign_y,
            trial_curl_sign_z,
        )
        stamp = _metal_gather_stage!("fused_pairs", timed, stamp)
        _metal_launch(
            _metal_fused_rhs_gather_kernel!,
            cache.p1_dof_count * chunk_count,
            rhs_partial,
            tables.blocks,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            cache.element_dp0_dofs,
            q_neumann,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            Int32(chunk_size),
            pair_stride,
            p1_count,
            dp0_count,
            drive_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_rhs", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        _metal_launch(
            _metal_fused_lhs_gather_kernel!,
            cache.p1_dof_count * node_count,
            lhs,
            tables.blocks,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            tables.chunk_nodes,
            tables.inc_offsets,
            tables.inc_packed,
            Int32(node_start),
            Int32(node_count),
            Int32(element_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_lhs", timed, stamp)
    end
    return nothing
end

# Singular corrections under fusion. The Duffy/Sauter-Schwab deltas are four
# per-operator blocks in the four-operator path; here they are combined with
# the same eta before the scatter. Getting this wrong is silent, so the
# equivalence gate compares a fused assembly against a four-operator one on the
# same mesh and frequency rather than trusting the validation tolerances.
function _metal_singular_fused_bm_blocks_kernel!(
    lhs_values,
    rhs_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    coupling_scale,
    face_count::Int32,
    pair_count::Int32,
    rule_point_count::Int32,
    part_count::Int32,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > pair_count * part_count && return nothing
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im = _metal_singular_pair_blocks(
        linear_index,
        test_indices, trial_indices, rule_indices, jac_scales, normal_products,
        rule_offsets, rule_test_points, rule_trial_points, rule_weights,
        face_vertices, normals, curls, k, face_count, pair_count, rule_point_count, part_count,
        trial_sign_x, trial_sign_y, trial_sign_z,
        trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    lhs_re, lhs_im, rhs_re, rhs_im = _metal_fused_pair_combination(
        slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im, coupling_scale,
    )
    value_stride = pair_count * part_count
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

function _metal_singular_fused_bm_scatter_kernel!(
    lhs_f32,
    rhs_f32,
    lhs_values,
    rhs_values,
    q_neumann,
    test_indices,
    trial_indices,
    p1_dofs,
    element_dp0_dofs,
    pair_count,
    part_count,
    p1_dof_count,
    dp0_dof_count,
    drive_count,
    face_count,
)
    pair_position = _metal_global_linear_index()
    pair_position > pair_count && return nothing
    value_stride = pair_count * part_count
    test_index = Int(test_indices[pair_position])
    trial_index = Int(trial_indices[pair_position])
    dp0_column = Int(element_dp0_dofs[trial_index])
    local_row = 1
    while local_row <= 3
        row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
        coefficient = zero(eltype(rhs_values))
        part = 1
        while part <= part_count
            coefficient += rhs_values[pair_position + (part - 1) * pair_count + (local_row - 1) * value_stride]
            part += 1
        end
        drive = 1
        while drive <= drive_count
            contribution = coefficient * q_neumann[dp0_column + (drive - 1) * dp0_dof_count]
            _metal_atomic_add_complex!(
                rhs_f32,
                row + (drive - 1) * p1_dof_count,
                real(contribution),
                imag(contribution),
            )
            drive += 1
        end
        local_row += 1
    end
    local_column = 1
    while local_column <= 3
        column = Int(p1_dofs[trial_index + (local_column - 1) * face_count])
        local_row = 1
        while local_row <= 3
            row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
            component = (local_column - 1) * 3 + local_row - 1
            value = zero(eltype(lhs_values))
            part = 1
            while part <= part_count
                value += lhs_values[pair_position + (part - 1) * pair_count + component * value_stride]
                part += 1
            end
            _metal_atomic_add_complex!(
                lhs_f32,
                row + (column - 1) * p1_dof_count,
                real(value),
                imag(value),
            )
            local_row += 1
        end
        local_column += 1
    end
    return nothing
end

function _launch_metal_fused_singular_kernels!(
    lhs,
    rhs,
    q_neumann,
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    k,
    coupling_scale,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    T = typeof(k)
    sx = T(transform.signs[1])
    sy = T(transform.signs[2])
    sz = T(transform.signs[3])
    csx = T(transform.determinant * transform.signs[1])
    csy = T(transform.determinant * transform.signs[2])
    csz = T(transform.determinant * transform.signs[3])
    rule_point_count = length(singular_cache.rule_weights)
    part_count = _metal_singular_part_count()
    value_count = pair_count * part_count
    lhs_values = Metal.zeros(eltype(lhs), value_count, 9)
    rhs_values = Metal.zeros(eltype(lhs), value_count, 3)
    _metal_launch(
        _metal_singular_fused_bm_blocks_kernel!,
        value_count,
        lhs_values, rhs_values,
        singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
        singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
        singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
        regular_cache.face_vertices, regular_cache.normals, regular_cache.curls,
        k, coupling_scale, Int32(regular_cache.face_count), Int32(pair_count),
        Int32(rule_point_count), Int32(part_count),
        sx, sy, sz, csx, csy, csz,
    )
    _metal_launch(
        _metal_singular_fused_bm_scatter_kernel!,
        pair_count,
        reinterpret(T, lhs),
        reinterpret(T, rhs),
        lhs_values,
        rhs_values,
        q_neumann,
        singular_cache.test_indices,
        singular_cache.trial_indices,
        regular_cache.p1_dofs,
        regular_cache.element_dp0_dofs,
        pair_count,
        part_count,
        regular_cache.p1_dof_count,
        regular_cache.dp0_dof_count,
        size(q_neumann, 2),
        regular_cache.face_count,
    )
    Metal.synchronize()
    Metal.unsafe_free!(lhs_values)
    Metal.unsafe_free!(rhs_values)
    return nothing
end

"""
    build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, T)

Device-side form of the two L2 identity blocks for the fused path. Both are
assembled dense but are structurally sparse (a P1 mass matrix), and both are
frequency-independent, so a sweep builds this once. The blocks already carry
the p1 symmetry orbit weights from `assemble_l2_identity_matrix`, which is why
the fused assembly weights only the operator part before adding them.
"""
struct MetalFusedIdentityCache{S,M}
    p1_p1_scatter::S
    p1_dp0::M
end

function build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, ::Type{T}) where {T<:AbstractFloat}
    _require_metal!()
    return MetalFusedIdentityCache(
        build_metal_sparse_scatter_cache(sparse(Complex{T}.(identity_p1_p1))),
        sparse(Complex{T}.(identity_p1_dp0)),
    )
end

function release_metal_fused_identity_cache!(cache::MetalFusedIdentityCache)
    release_metal_sparse_scatter_cache!(cache.p1_p1_scatter)
    return nothing
end

"""
    assemble_burton_miller_neumann_system_metal(mesh, p1_space, dp0_space, q_neumann, k, rule; ...)

Assemble the Burton-Miller Neumann system directly, without ever forming S,
K', D or H. `q_neumann` is `dp0_dof_count x drive_count`; every drive's
right-hand side is accumulated in the same pass, so one assembly serves the
whole channel set at a frequency exactly as one factorization does.

Returns `(matrix, rhs, metadata)` with `matrix` `N x N` and `rhs` `N x drives`,
both Metal-resident. The caller owns and releases them.
"""
function assemble_burton_miller_neumann_system_metal(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann,
    k::T,
    rule::TriangleRule{T};
    device_cache,
    singular_cache=nothing,
    device_singular_cache=nothing,
    identity_p1_p1=nothing,
    identity_p1_dp0=nothing,
    identity_cache=nothing,
    skip_singular::Bool=false,
    singular_order::Int=4,
    element_indices=eachindex(mesh.faces),
    symmetry_mode::Symbol=:off,
    timing=nothing,
    coupling_cap::Real=zero(T),
) where {T<:AbstractFloat}
    _require_metal!()
    device_cache isa MetalRegularAssemblyCache ||
        error("Fused Metal Burton-Miller assembly requires a MetalRegularAssemblyCache.")
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    device_cache.symmetry_mode == normalized_mode ||
        error("Metal assembly cache symmetry mode $(device_cache.symmetry_mode) does not match requested $(normalized_mode).")
    q_host = q_neumann isa AbstractMatrix ? q_neumann : reshape(q_neumann, :, 1)
    size(q_host, 1) == dp0_space.global_dof_count ||
        error("Fused Metal Burton-Miller assembly needs one Neumann row per DP0 dof.")
    drive_count = size(q_host, 2)
    drive_count >= 1 || error("Fused Metal Burton-Miller assembly needs at least one drive.")
    p1_count = p1_space.global_dof_count

    owns_identity_cache = identity_cache === nothing
    if owns_identity_cache
        (identity_p1_p1 === nothing || identity_p1_dp0 === nothing) &&
            error("Fused Metal Burton-Miller assembly needs identity_cache or both identity blocks.")
        identity_cache = build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, T)
    end
    coupling_scale = burton_miller_coupling_scale(k, coupling_cap)
    d_q = q_host isa MtlArray ? q_host : MtlArray(Complex{T}.(q_host))
    owns_q = !(q_host isa MtlArray)
    tables = _metal_fused_gather_tables(device_cache)
    lhs = rhs = rhs_partial = nothing
    succeeded = false
    empty!(_metal_gather_stage_timing)
    try
        storage = metal_operator_storage_mode()
        allocation_elapsed = @elapsed begin
            # The system matrix and right-hand side go to the same storage mode
            # as the four operators, so the host can wrap them in place instead
            # of blitting them through a staging buffer. The pair-block and
            # right-hand-side partials are device-only scratch and stay private.
            lhs = Metal.zeros(Complex{T}, p1_count, p1_count; storage=storage)
            rhs = Metal.zeros(Complex{T}, p1_count, drive_count; storage=storage)
            rhs_partial = Metal.zeros(Complex{T}, p1_count, tables.chunk_size, drive_count)
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_fused_alloc"] = allocation_elapsed)

        singular_mode = _normalized_metal_singular_mode()
        singular_mode == :native ||
            error("Fused Metal Burton-Miller assembly has no host singular mode; unset BLAB_METAL_SINGULAR_MODE.")
        skip_image_singular = !skip_singular
        one_t = one(T)
        kernel_elapsed = @elapsed begin
            _launch_metal_fused_pair_kernels!(
                lhs, rhs_partial, d_q, device_cache, k, coupling_scale,
                device_cache.vertex_offsets, device_cache.incident_elements, Int32(0),
                one_t, one_t, one_t, one_t, one_t, one_t,
            )
            for (transform, image_cache) in zip(device_cache.image_transforms, device_cache.image_singular_caches)
                _launch_metal_fused_pair_kernels!(
                    lhs, rhs_partial, d_q, device_cache, k, coupling_scale,
                    image_cache.pair_offsets, image_cache.trial_indices,
                    skip_image_singular ? Int32(1) : Int32(2),
                    T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                    T(transform.determinant * transform.signs[1]),
                    T(transform.determinant * transform.signs[2]),
                    T(transform.determinant * transform.signs[3]),
                )
            end
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_fused_regular_kernel"] = kernel_elapsed)
        if timing !== nothing
            for (stage, elapsed) in _metal_gather_stage_timing
                timing["metal_fused_" * stage] = elapsed
            end
        end

        reduce_elapsed = @elapsed begin
            _metal_launch(
                _metal_fused_rhs_reduce_kernel!,
                p1_count * drive_count,
                rhs, rhs_partial,
                Int32(p1_count), Int32(tables.chunk_size), Int32(drive_count),
            )
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_fused_rhs_reduce"] = reduce_elapsed)

        indices = device_cache.element_indices
        correction_cache = singular_cache === nothing ?
            build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
        singular_pairs = 0
        image_singular_pairs = 0
        if !skip_singular
            owns_device_singular_cache = device_singular_cache === nothing
            active_singular_cache = device_singular_cache === nothing ?
                build_metal_singular_correction_cache(correction_cache) : device_singular_cache
            singular_elapsed = @elapsed begin
                _launch_metal_fused_singular_kernels!(lhs, rhs, d_q, device_cache, active_singular_cache, k, coupling_scale)
            end
            timing !== nothing && (timing["metal_fused_singular_kernel"] = singular_elapsed)
            singular_pairs = correction_cache.pair_count
            owns_device_singular_cache && release_metal_singular_correction_cache!(active_singular_cache)
            image_elapsed = @elapsed begin
                for (transform, image_cache) in zip(device_cache.image_transforms, device_cache.image_singular_caches)
                    image_cache.pair_count == 0 && continue
                    _launch_metal_fused_singular_kernels!(lhs, rhs, d_q, device_cache, image_cache, k, coupling_scale, transform)
                end
                Metal.synchronize()
            end
            timing !== nothing && (timing["metal_fused_image_singular_kernel"] = image_elapsed)
            image_singular_pairs = device_cache.image_singular_pair_count
        end

        # Row weights scale the operator part only, exactly as the four-operator
        # path scales S, K', D and H before the host adds the identity blocks.
        weight_elapsed = @elapsed begin
            if normalized_mode != :off
                d_weights = MtlArray(Complex{T}.(p1_symmetry_orbit_weights(mesh, normalized_mode)))
                lhs .*= reshape(d_weights, :, 1)
                rhs .*= reshape(d_weights, :, 1)
                Metal.synchronize()
                Metal.unsafe_free!(d_weights)
            end
        end
        timing !== nothing && (timing["metal_fused_symmetry_row_weights"] = weight_elapsed)

        # lhs += 0.5 I_p1p1 ; rhs += -0.5 (i/k) I_p1dp0 q. The identity blocks
        # are sparse and the right-hand-side term is one sparse matvec per
        # drive, so both stay cheaper on the host than a kernel launch.
        identity_elapsed = @elapsed begin
            scatter_metal_sparse_to_dense!(lhs, identity_cache.p1_p1_scatter; alpha=Complex{T}(0.5), add=true)
            coupling = Complex{T}(zero(T), coupling_scale)
            identity_rhs = (identity_cache.p1_dp0 * Complex{T}.(q_host)) .* (-Complex{T}(0.5) * coupling)
            d_identity_rhs = MtlArray(identity_rhs)
            try
                rhs .+= d_identity_rhs
                Metal.synchronize()
            finally
                Metal.unsafe_free!(d_identity_rhs)
            end
        end
        timing !== nothing && (timing["metal_fused_identity"] = identity_elapsed)

        total_pairs = length(indices) * length(indices)
        image_count = length(device_cache.image_transforms)
        succeeded = true
        return (
            matrix=lhs,
            rhs=rhs,
            regular_pairs=total_pairs - correction_cache.pair_count +
                image_count * total_pairs - image_singular_pairs,
            singular_pairs=singular_pairs,
            image_singular_pairs=image_singular_pairs,
            drive_count=drive_count,
            on_gpu=true,
            gpu_backend=:metal,
            assembly_mode=:metal_fused_burton_miller,
        )
    finally
        rhs_partial === nothing || Metal.unsafe_free!(rhs_partial)
        owns_q && Metal.unsafe_free!(d_q)
        owns_identity_cache && release_metal_fused_identity_cache!(identity_cache)
        if !succeeded
            lhs === nothing || Metal.unsafe_free!(lhs)
            rhs === nothing || Metal.unsafe_free!(rhs)
        end
    end
end

function release_metal_burton_miller_system!(system)
    backing = get(system, :metal_backing, nothing)
    if backing !== nothing
        Metal.unsafe_free!(backing.matrix)
        Metal.unsafe_free!(backing.rhs)
        return nothing
    end
    get(system, :on_gpu, false) || return nothing
    Metal.unsafe_free!(system.matrix)
    Metal.unsafe_free!(system.rhs)
    return nothing
end

"""
    metal_host_burton_miller_system(system)

Present the fused system to the host. Shared-storage buffers are wrapped in
place, so this costs nothing and the returned arrays alias device memory; the
returned tuple carries the device arrays under `metal_backing` and owns them,
exactly as `metal_host_operators` does for the four-operator path. Release
once, through whichever tuple you still hold.
"""
function metal_host_burton_miller_system(system)
    get(system, :on_gpu, false) || return system
    Metal.synchronize()
    shared = Metal.is_shared(system.matrix) && Metal.is_shared(system.rhs)
    host = shared ?
        (matrix=unsafe_wrap(Array, system.matrix), rhs=unsafe_wrap(Array, system.rhs)) :
        (matrix=Array(system.matrix), rhs=Array(system.rhs))
    backing = shared ? (matrix=system.matrix, rhs=system.rhs) : nothing
    extras = Base.structdiff(system, NamedTuple{(:matrix, :rhs, :on_gpu, :metal_backing)})
    # Private storage leaves the device tuple the caller's to free; shared
    # storage hands ownership of the wrapped buffers to the returned tuple.
    return merge(extras, host, (on_gpu=false, metal_backing=backing))
end

"""
    solve_metal_burton_miller_system_with_report(system; method=beat_dense_solve_method())

Solve the fused system on the host -- Metal.jl has no GPU LU, and the shared
storage the assembly uses means the host reads the device buffers in place
rather than copying them. Dense LU or diagonally preconditioned GMRES is
chosen by cost model over (dofs, drives); see `beat_solve_dense_system`.

Returns `(pressure, report)` with pressure `N x drives`. The system is left
allocated; the caller releases it.

The LU route factors once and solves every drive against that one
factorization, which is the property the CPU and Metal backends were built to
have. GMRES has no factorization to share and pays per drive, which is exactly
what the router weighs -- and why it is a cost comparison over both dimensions
rather than a dof threshold.
"""
function solve_metal_burton_miller_system_with_report(system; method::Symbol=beat_dense_solve_method())
    host = metal_host_burton_miller_system(system)
    # lu! would overwrite the shared buffer the caller still owns; GMRES reads
    # it and needs no copy at all.
    return beat_solve_dense_system(host.matrix, host.rhs; method=method, preserve_matrix=true)
end

function solve_metal_burton_miller_system(system; method::Symbol=beat_dense_solve_method())
    pressure, _ = solve_metal_burton_miller_system_with_report(system; method=method)
    return pressure
end
