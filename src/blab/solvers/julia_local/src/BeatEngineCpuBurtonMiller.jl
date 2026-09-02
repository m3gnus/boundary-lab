# Fused Burton-Miller exterior assembly for the CPU backend.
#
# Same change as the Metal path in BeatEngineMetalBurtonMiller.jl, and for the
# same reasons: the coupling eta = i |k| / max(k^2, cap) is known at assembly
# time, so
#
#   lhs = 0.5 I_p1p1 - D + eta H            rhs = (-S - eta (K' + 0.5 I)) q
#
# can be formed per element pair instead of assembling S, K', D and H and
# combining them afterwards. The result is one N x N matrix and one right-hand
# side per drive rather than 6N^2 complex entries.
#
# The combination is applied per quadrature point pair, before the 3x3
# accumulation, not afterwards to four finished blocks. That is where the time
# is: combining afterwards was measured at 1.00x on the Metal pair kernel,
# combining before it at 1.85x. Per entry of the 3x3 block the four-operator
# form costs a multiply, a subtract and two FMAs across two accumulators; the
# combined form costs one FMA into one, because the hypersingular curl term
# carries no basis product and can be summed once and added after the loop.
#
# The four-operator path stays for the coupled FEM/LEM solver and for any
# operator preconditioner. This is an exterior-only fast path.
#
# Every stage is test-element owned exactly as the four-operator assembly is,
# so the same colouring keeps the threaded writes race-free.

@inline function _beat_cpu_bm_scatter!(
    lhs,
    rhs,
    q_neumann,
    test_p1_dofs::NTuple{3,Int},
    trial_p1_dofs::NTuple{3,Int},
    dp0_dof::Int,
    lhs_block,
    rhs_block,
    ::Val{subtract},
) where {subtract}
    drive_count = size(q_neumann, 2)
    @inbounds for local_row in 1:3
        row = test_p1_dofs[local_row]
        coefficient = subtract ? -rhs_block[local_row] : rhs_block[local_row]
        for drive in 1:drive_count
            rhs[row, drive] += coefficient * q_neumann[dp0_dof, drive]
        end
        for local_col in 1:3
            column = trial_p1_dofs[local_col]
            value = lhs_block[local_row, local_col]
            lhs[row, column] += subtract ? -value : value
        end
    end
    return nothing
end

# The pair's Burton-Miller contribution, combined per quadrature point pair.
#
#   lhs entry = -D + eta H
#             = sum_q basis_product * (-wd - eta k^2 (n.n') wg) + curl * eta sum_q wg
#   rhs entry = -S - eta K' = sum_q test_basis * (-wg - eta wa)
#
# The curl term carries no basis product, so it leaves the 3x3 loop entirely:
# one scalar accumulates over the quadrature and the 3x3 outer product is added
# once after it. What is left inside is one FMA per entry.
function _beat_cpu_bm_regular_pair_blocks(
    test_data::BeatCpuElementData{T},
    trial_data::BeatCpuElementData{T},
    test_quad::BeatCpuRegularQuadratureData{T},
    trial_quad::BeatCpuRegularQuadratureData{T},
    normal_product::T,
    jac_scale::T,
    k::T,
    coupling::Complex{T},
) where {T<:AbstractFloat}
    lhs_block = zero(MMatrix{3,3,Complex{T},9})
    rhs_block = zero(MVector{3,Complex{T}})
    curl_total = zero(Complex{T})
    k2n = coupling * (k * k * normal_product)

    @inbounds for test_q in eachindex(test_quad.weights)
        test_basis = test_quad.basis[test_q]
        x = test_quad.points[test_q]
        test_weight = test_quad.weights[test_q]
        for trial_q in eachindex(trial_quad.weights)
            trial_basis = trial_quad.basis[trial_q]
            r_vec = trial_quad.points[trial_q] - x
            radius = norm(r_vec)
            radius == zero(T) && continue
            inv_radius = inv(radius)
            green = _beat_cpu_green(radius, inv_radius, k)
            grad_scale = green * Complex{T}(-inv_radius, k)
            weight = test_weight * trial_quad.weights[trial_q] * jac_scale
            weighted_green = green * weight
            lhs_scale = -grad_scale * (dot(r_vec, trial_data.normal) * inv_radius) * weight -
                k2n * weighted_green
            rhs_scale = -weighted_green -
                coupling * grad_scale * (-dot(r_vec, test_data.normal) * inv_radius) * weight
            curl_total += weighted_green
            for local_row in 1:3
                test_value = test_basis[local_row]
                rhs_block[local_row] += test_value * rhs_scale
                for local_col in 1:3
                    lhs_block[local_row, local_col] +=
                        (test_value * trial_basis[local_col]) * lhs_scale
                end
            end
        end
    end
    curl_total *= coupling
    @inbounds for local_row in 1:3, local_col in 1:3
        lhs_block[local_row, local_col] +=
            dot(test_data.curls[local_row], trial_data.curls[local_col]) * curl_total
    end
    return lhs_block, rhs_block
end

# The same combination over a Duffy (or image-delta) rule, whose points and
# basis values are not precomputed per element.
function _beat_cpu_bm_pair_blocks(
    test_vertices::NTuple{3,SVector{3,T}},
    trial_vertices::NTuple{3,SVector{3,T}},
    test_normal::SVector{3,T},
    trial_normal::SVector{3,T},
    test_curls::NTuple{3,SVector{3,T}},
    trial_curls::NTuple{3,SVector{3,T}},
    normal_product::T,
    jac_scale::T,
    k::T,
    coupling::Complex{T},
    test_points,
    trial_points,
    weights,
) where {T<:AbstractFloat}
    lhs_block = zero(MMatrix{3,3,Complex{T},9})
    rhs_block = zero(MVector{3,Complex{T}})
    curl_total = zero(Complex{T})
    k2n = coupling * (k * k * normal_product)

    @inbounds for q in eachindex(weights)
        test_basis = p1_values(test_points[q])
        trial_basis = p1_values(trial_points[q])
        x = local_to_global(test_vertices, test_points[q])
        y = local_to_global(trial_vertices, trial_points[q])
        r_vec = y - x
        radius = norm(r_vec)
        radius == zero(T) && continue
        inv_radius = inv(radius)
        green = _beat_cpu_green(radius, inv_radius, k)
        grad_scale = green * Complex{T}(-inv_radius, k)
        weight = weights[q] * jac_scale
        weighted_green = green * weight
        lhs_scale = -grad_scale * (dot(r_vec, trial_normal) * inv_radius) * weight -
            k2n * weighted_green
        rhs_scale = -weighted_green -
            coupling * grad_scale * (-dot(r_vec, test_normal) * inv_radius) * weight
        curl_total += weighted_green
        for local_row in 1:3
            test_value = test_basis[local_row]
            rhs_block[local_row] += test_value * rhs_scale
            for local_col in 1:3
                lhs_block[local_row, local_col] += (test_value * trial_basis[local_col]) * lhs_scale
            end
        end
    end
    curl_total *= coupling
    @inbounds for local_row in 1:3, local_col in 1:3
        lhs_block[local_row, local_col] += dot(test_curls[local_row], trial_curls[local_col]) * curl_total
    end
    return lhs_block, rhs_block
end

function _beat_cpu_bm_regular_test!(
    lhs, rhs, q_neumann, elements, test_index::Int, trial_indices, k::T,
    regular_quadrature, coupling::Complex{T},
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]
    for trial_index in trial_indices
        trial_data = elements[trial_index]
        elements_are_adjacent(test_data.face, trial_data.face) && continue
        lhs_block, rhs_block = _beat_cpu_bm_regular_pair_blocks(
            test_data, trial_data, test_quad, regular_quadrature[trial_index],
            dot(test_data.normal, trial_data.normal),
            T(4.0) * test_data.area * trial_data.area, k, coupling,
        )
        _beat_cpu_bm_scatter!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            lhs_block, rhs_block, Val(false),
        )
    end
    return nothing
end

function _beat_cpu_bm_regular_image_test!(
    lhs, rhs, q_neumann, elements, image_elements, test_index::Int, trial_indices, k::T,
    regular_quadrature, image_quadrature, coupling::Complex{T},
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]
    for trial_index in trial_indices
        trial_data = image_elements[trial_index]
        lhs_block, rhs_block = _beat_cpu_bm_regular_pair_blocks(
            test_data, trial_data, test_quad, image_quadrature[trial_index],
            dot(test_data.normal, trial_data.normal),
            T(4.0) * test_data.area * trial_data.area, k, coupling,
        )
        _beat_cpu_bm_scatter!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            lhs_block, rhs_block, Val(false),
        )
    end
    return nothing
end

function _beat_cpu_bm_singular_test!(
    lhs, rhs, q_neumann, elements, pairs, rules, k::T, coupling::Complex{T},
) where {T<:AbstractFloat}
    for pair in pairs
        test_data = elements[pair.test_index]
        trial_data = elements[pair.trial_index]
        duffy = rules[pair.rule_index]
        lhs_block, rhs_block = _beat_cpu_bm_pair_blocks(
            test_data.vertices, trial_data.vertices, test_data.normal, trial_data.normal,
            test_data.curls, trial_data.curls, pair.normal_product, pair.jac_scale, k, coupling,
            duffy.test_points, duffy.trial_points, duffy.weights,
        )
        _beat_cpu_bm_scatter!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            lhs_block, rhs_block, Val(false),
        )
    end
    return nothing
end

# Image-singular correction: the Duffy value minus the regular-rule value the
# image pass already added, combined with the same eta before the scatter. The
# four-operator path applies four separate deltas here; folding them is the
# most likely place for a silent sign error, which is why the equivalence gate
# runs the symmetry fixtures.
function _beat_cpu_bm_image_singular_delta_test!(
    lhs, rhs, q_neumann, elements, image_elements, pairs, rules, k::T,
    regular_quadrature, image_quadrature, coupling::Complex{T},
) where {T<:AbstractFloat}
    for pair in pairs
        test_data = elements[pair.test_index]
        trial_data = image_elements[pair.trial_index]
        duffy = rules[pair.rule_index]
        lhs_block, rhs_block = _beat_cpu_bm_pair_blocks(
            test_data.vertices, trial_data.vertices, test_data.normal, trial_data.normal,
            test_data.curls, trial_data.curls, pair.normal_product, pair.jac_scale, k, coupling,
            duffy.test_points, duffy.trial_points, duffy.weights,
        )
        _beat_cpu_bm_scatter!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            lhs_block, rhs_block, Val(false),
        )
        regular_lhs, regular_rhs = _beat_cpu_bm_regular_pair_blocks(
            test_data, trial_data, regular_quadrature[pair.test_index],
            image_quadrature[pair.trial_index], pair.normal_product,
            T(4.0) * test_data.area * trial_data.area, k, coupling,
        )
        _beat_cpu_bm_scatter!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            regular_lhs, regular_rhs, Val(true),
        )
    end
    return nothing
end

"""
    assemble_burton_miller_neumann_system_cpu(mesh, p1_space, dp0_space, q_neumann, k, rule; ...)

Assemble the Burton-Miller Neumann system directly on the CPU, without forming
S, K', D or H. `q_neumann` is `dp0_dof_count x drive_count`; every drive's
right-hand side is accumulated in the same pass.

Returns `(matrix, rhs, ...)` with `matrix` `N x N` and `rhs` `N x drives`.
"""
function assemble_burton_miller_neumann_system_cpu(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann,
    k::T,
    rule::TriangleRule{T};
    identity_p1_p1,
    identity_p1_dp0,
    skip_singular::Bool=false,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    threaded::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    cpu_cache=nothing,
    symmetry_mode::Symbol=:off,
    coupling_cap::Real=zero(T),
) where {T<:AbstractFloat}
    symmetry_mode = normalized_symmetry_mode(symmetry_mode)
    if cpu_cache !== nothing
        cpu_cache.symmetry_mode == symmetry_mode || error("CPU assembly cache symmetry mode does not match the requested mode.")
        cpu_cache.singular_order == singular_order || error("CPU assembly cache singular order does not match the requested order.")
        cpu_cache.rule == rule || error("CPU assembly cache quadrature rule does not match the requested rule.")
    end
    q_host = q_neumann isa AbstractMatrix ? q_neumann : reshape(q_neumann, :, 1)
    size(q_host, 1) == dp0_space.global_dof_count ||
        error("Fused CPU Burton-Miller assembly needs one Neumann row per DP0 dof.")
    q_complex = Complex{T}.(q_host)
    drive_count = size(q_complex, 2)

    indices = cpu_cache === nothing ? collect(element_indices) : cpu_cache.indices
    p1_count = p1_space.global_dof_count
    lhs = zeros(Complex{T}, p1_count, p1_count)
    rhs = zeros(Complex{T}, p1_count, drive_count)
    coupling = burton_miller_coupling(k, coupling_cap)
    elements = cpu_cache === nothing ? _beat_cpu_element_data(mesh, p1_space, dp0_space) : cpu_cache.elements
    regular_quadrature = cpu_cache === nothing ? _beat_cpu_regular_quadrature_data(mesh, rule) : cpu_cache.regular_quadrature
    adjacent_pairs = cpu_cache === nothing ? count_adjacent_pairs(mesh, indices) : cpu_cache.adjacent_pairs
    threaded_enabled = cpu_cache === nothing ? threaded && Threads.nthreads() > 1 : cpu_cache.threaded_enabled
    color_groups = if cpu_cache === nothing
        threaded_enabled ? _beat_cpu_element_color_groups(mesh, indices) : [indices]
    else
        cpu_cache.color_groups
    end
    image_transforms = cpu_cache === nothing ? collect(symmetry_image_transforms(symmetry_mode)) : cpu_cache.image_transforms

    regular_elapsed = @elapsed begin
        if threaded_enabled
            for group in color_groups
                Threads.@threads for group_index in eachindex(group)
                    _beat_cpu_bm_regular_test!(
                        lhs, rhs, q_complex, elements, group[group_index], indices, k,
                        regular_quadrature, coupling,
                    )
                end
            end
        else
            for test_index in indices
                _beat_cpu_bm_regular_test!(
                    lhs, rhs, q_complex, elements, test_index, indices, k,
                    regular_quadrature, coupling,
                )
            end
        end
        for (transform_index, transform) in enumerate(image_transforms)
            image_elements = cpu_cache === nothing ?
                _beat_cpu_reflect_element_data(elements, transform) :
                cpu_cache.image_elements[transform_index]
            image_quadrature = cpu_cache === nothing ?
                _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) :
                cpu_cache.image_quadrature[transform_index]
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        _beat_cpu_bm_regular_image_test!(
                            lhs, rhs, q_complex, elements, image_elements, group[group_index],
                            indices, k, regular_quadrature, image_quadrature, coupling,
                        )
                    end
                end
            else
                for test_index in indices
                    _beat_cpu_bm_regular_image_test!(
                        lhs, rhs, q_complex, elements, image_elements, test_index,
                        indices, k, regular_quadrature, image_quadrature, coupling,
                    )
                end
            end
        end
    end
    timing !== nothing && (timing["fused_regular_cpu_scatter"] = regular_elapsed)

    singular_pairs = 0
    if !skip_singular
        cache = singular_cache === nothing ? build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
        singular_elapsed = @elapsed begin
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        _beat_cpu_bm_singular_test!(
                            lhs, rhs, q_complex, elements,
                            cache.pairs_by_test[group[group_index]], cache.rules, k, coupling,
                        )
                    end
                end
            else
                for test_index in indices
                    _beat_cpu_bm_singular_test!(
                        lhs, rhs, q_complex, elements,
                        cache.pairs_by_test[test_index], cache.rules, k, coupling,
                    )
                end
            end
        end
        timing !== nothing && (timing["fused_singular_cpu_scatter"] = singular_elapsed)
        singular_pairs = cache.pair_count
    end

    image_singular_pairs = 0
    image_singular_elapsed = @elapsed begin
        if !skip_singular
            for (transform_index, transform) in enumerate(image_transforms)
                image_cache = cpu_cache === nothing ?
                    _beat_cpu_image_singular_cache(mesh, singular_order, indices, transform) :
                    cpu_cache.image_singular_caches[transform_index]
                image_singular_pairs += image_cache.pair_count
                image_cache.pair_count == 0 && continue
                image_elements = cpu_cache === nothing ?
                    _beat_cpu_reflect_element_data(elements, transform) :
                    cpu_cache.image_elements[transform_index]
                image_quadrature = cpu_cache === nothing ?
                    _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) :
                    cpu_cache.image_quadrature[transform_index]
                if threaded_enabled
                    for group in color_groups
                        Threads.@threads for group_index in eachindex(group)
                            _beat_cpu_bm_image_singular_delta_test!(
                                lhs, rhs, q_complex, elements, image_elements,
                                image_cache.pairs_by_test[group[group_index]], image_cache.rules,
                                k, regular_quadrature, image_quadrature, coupling,
                            )
                        end
                    end
                else
                    for test_index in indices
                        _beat_cpu_bm_image_singular_delta_test!(
                            lhs, rhs, q_complex, elements, image_elements,
                            image_cache.pairs_by_test[test_index], image_cache.rules,
                            k, regular_quadrature, image_quadrature, coupling,
                        )
                    end
                end
            end
        end
    end
    timing !== nothing && (timing["fused_image_singular_cpu_scatter"] = image_singular_elapsed)

    # Row weights scale the operator part only: `assemble_l2_identity_matrix`
    # already applies them to both identity blocks.
    weights = p1_symmetry_orbit_weights(mesh, symmetry_mode)
    if symmetry_mode != :off
        lhs .*= reshape(Complex{T}.(weights), :, 1)
        rhs .*= reshape(Complex{T}.(weights), :, 1)
    end
    identity_elapsed = @elapsed begin
        lhs .+= Complex{T}(0.5) .* Complex{T}.(identity_p1_p1)
        rhs .+= (Complex{T}.(identity_p1_dp0) * q_complex) .* (-Complex{T}(0.5) * coupling)
    end
    timing !== nothing && (timing["fused_identity_cpu"] = identity_elapsed)

    return (
        matrix=lhs,
        rhs=rhs,
        regular_pairs=length(indices) * length(indices) - adjacent_pairs +
            length(image_transforms) * length(indices) * length(indices) - image_singular_pairs,
        singular_pairs=singular_pairs,
        image_singular_pairs=image_singular_pairs,
        drive_count=drive_count,
        on_gpu=false,
        assembly_mode=:cpu_fused_burton_miller,
    )
end

"""
    solve_burton_miller_neumann_system_cpu_with_report(system; method=beat_dense_solve_method())

Solve the fused CPU system, choosing dense LU or diagonally preconditioned
GMRES by cost model. Returns `(pressure, report)`; see
`beat_solve_dense_system`.

The matrix is preserved: `lu!` would overwrite it, and the caller may hold it
for a diagnostic or a second solve. GMRES never writes to it, so the GMRES
route also skips the copy the LU route needs.
"""
function solve_burton_miller_neumann_system_cpu_with_report(system; method::Symbol=beat_dense_solve_method())
    return beat_solve_dense_system(system.matrix, system.rhs; method=method, preserve_matrix=true)
end

function solve_burton_miller_neumann_system_cpu(system; method::Symbol=beat_dense_solve_method())
    pressure, _ = solve_burton_miller_neumann_system_cpu_with_report(system; method=method)
    return pressure
end
