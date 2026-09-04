using Test
using StaticArrays
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

cuda_available() = CUDA_MODULE !== nothing && CUDA_MODULE.functional()

const AMDGPU_MODULE = try
    @eval import AMDGPU
    AMDGPU
catch
    nothing
end

rocm_available() = AMDGPU_MODULE !== nothing &&
                   AMDGPU_MODULE.functional() &&
                   AMDGPU_MODULE.functional(:rocblas) &&
                   AMDGPU_MODULE.functional(:rocsolver)

@testset "symmetry plane snapping" begin
    vertices = [
        SVector{3,Float64}(-1.2e-8, 0.0, 0.8),
        SVector{3,Float64}(0.5, 0.0, 0.8),
        SVector{3,Float64}(0.0, 0.5, 0.8),
    ]
    mesh = BoundaryMesh(vertices, [(1, 2, 3)], [1])

    tolerance = symmetry_plane_tolerance(mesh.vertices)
    snapped = snap_symmetry_planes(mesh, :x)

    @test tolerance ≈ sqrt(0.5) * 1.0e-6
    @test snapped.vertices[1][1] == 0.0
    @test mesh.vertices[1][1] == -1.2e-8
    validate_symmetry_fundamental_domain!(mesh, :x)
    @test_throws ErrorException validate_symmetry_fundamental_domain!(
        mesh,
        :x;
        tolerance=1.0e-9,
    )
end

@testset "mesh setup" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    singular_cache = build_singular_correction_cache(mesh, 2)

    @test length(mesh.faces) > 0
    @test length(mesh.vertices) > 0
    @test p1.global_dof_count == length(mesh.vertices)
    @test dp0.global_dof_count == length(mesh.faces)
    @test length(rule.points) == length(rule.weights)
    @test singular_cache.pair_count > 0
end

include(joinpath(@__DIR__, "coupled_solver_tests.jl"))
include(joinpath(@__DIR__, "coupled_condensed_tests.jl"))

@testset "cpu BLAS thread policy" begin
    @test beat_cpu_blas_thread_count(441; available_threads=16) == 1
    @test beat_cpu_blas_thread_count(1390; available_threads=16) == 4
    @test beat_cpu_blas_thread_count(3502; available_threads=16) == 8
    @test beat_cpu_blas_thread_count(5000; available_threads=16) == 16
    @test beat_cpu_blas_thread_count(3502; available_threads=4) == 4
    @test beat_cpu_blas_thread_count(441; available_threads=16, override="3") == 3
    @test_throws ErrorException beat_cpu_blas_thread_count(441; available_threads=16, override="invalid")
    @test_throws ErrorException beat_cpu_blas_thread_count(441; available_threads=16, override="0")
end

@testset "cpu production pipeline" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
    off_cache = build_beat_cpu_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=2,
        element_indices=element_indices,
        symmetry_mode=:off,
    )
    @test isempty(off_cache.image_transforms)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        backend=:cpu,
        singular_cache=singular_cache,
    )

    @test !get(operators, :on_gpu, true)
    expected_cpu_mode = Threads.nthreads() > 1 ? :cpu_colored_threads : :cpu_serial
    expected_cpu_kernel = Threads.nthreads() > 1 ? "cpu_colored_threads" : "cpu_serial"
    @test operators.regular_assembly_mode == expected_cpu_mode
    @test operators.regular_kernel_mode == expected_cpu_kernel
    @test operators.cpu_color_count >= 1
    @test operators.regular_pairs > 0
    @test operators.singular_pairs == singular_cache.pair_count
    @test sum(abs2, operators.single_layer) > 0
    @test sum(abs2, operators.double_layer) > 0
    @test sum(abs2, operators.adjoint_double_layer) > 0
    @test sum(abs2, operators.hypersingular) > 0
    @test all(isfinite, real.(operators.single_layer))
    @test all(isfinite, imag.(operators.single_layer))

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    q_neumann[1] = ComplexF32(0, 1)
    pressure = solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    solve_system = build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k)
    pressure_from_system = solve_burton_miller_neumann_cpu_system(solve_system, q_neumann, Float32)

    @test length(pressure) == p1.global_dof_count
    @test all(isfinite, real.(pressure))
    @test all(isfinite, imag.(pressure))
    @test pressure_from_system ≈ pressure rtol=Float32(1e-4) atol=Float32(1e-4)

    # The cached system builds its right-hand side from three matrix-vector
    # products instead of materialising the N x 2N Burton-Miller right-hand
    # operator. Pin it against that operator, and pin the fused left-hand side
    # against the promote-then-broadcast form it replaced.
    reference_lhs, reference_rhs_operator = burton_miller_neumann_matrices(
        operators, identity_p1_p1, identity_p1_dp0, k,
    )
    coupling = ComplexF32(0, 1) / k
    promoted_lhs = ComplexF32(0.5) .* ComplexF32.(identity_p1_p1) .- operators.double_layer .+
                   coupling .* operators.hypersingular
    @test burton_miller_neumann_lhs(operators, identity_p1_p1, k) == promoted_lhs
    @test reference_lhs == promoted_lhs
    @test burton_miller_neumann_rhs(operators, identity_p1_dp0, q_neumann, k) ≈
          reference_rhs_operator * ComplexF32.(q_neumann) rtol=Float32(1e-5) atol=Float32(1e-6)
    # A complex identity block takes the BLAS path; it must agree with the real one.
    @test burton_miller_neumann_rhs(operators, ComplexF32.(identity_p1_dp0), q_neumann, k) ≈
          reference_rhs_operator * ComplexF32.(q_neumann) rtol=Float32(1e-5) atol=Float32(1e-6)

    field_cache = build_field_evaluation_cache(mesh, rule)
    eval_points = fibonacci_sphere(8, Float32(2.0))
    field = evaluate_galerkin_field_cpu(eval_points, mesh, pressure, q_neumann, k, field_cache)
    @test length(field) == length(eval_points)
    @test all(isfinite, real.(field))
    @test all(isfinite, imag.(field))

end

@testset "cpu fused Burton-Miller equals the four-operator path" begin
    # The fused exterior path forms 0.5 I - D + (i/k) H and (-S - (i/k)(K' +
    # 0.5 I)) q inside the assembly instead of combining four operators on the
    # host. Both run in Float32, so they differ only by summation order. This
    # runs every symmetry mode because the image-singular delta is where a sign
    # slip in the fusion would be silent.
    for (mesh_name, symmetry_mode) in (
        ("sample.msh", :off),
        ("sample_half.msh", :x),
        ("sample_quarter.msh", :xy),
    )
        mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", mesh_name), Float32(0.001))
        mesh = snap_symmetry_planes(mesh, symmetry_mode)
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(Float32, 2)
        k = Float32(2pi * 1500.0 / 343.0)
        element_indices = 1:min(48, length(mesh.faces))
        singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
        cpu_cache = build_beat_cpu_assembly_cache(
            mesh, p1, dp0, rule;
            singular_order=2, element_indices=element_indices, symmetry_mode=symmetry_mode,
        )
        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)

        operators = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, element_indices=element_indices,
            backend=:cpu, singular_cache=singular_cache, cpu_cache=cpu_cache,
            symmetry_mode=symmetry_mode,
        )
        reference_lhs, reference_rhs_operator = BeatEngineCore.burton_miller_neumann_matrices(
            operators, identity_p1_p1, identity_p1_dp0, k,
        )

        drive_count = 3
        q_neumann = ComplexF32[
            ComplexF32(sin(Float32(0.7 * row + 1.3 * column)), cos(Float32(0.4 * row - 0.9 * column)))
            for row in 1:dp0.global_dof_count, column in 1:drive_count
        ]
        fused = assemble_burton_miller_neumann_system_cpu(
            mesh, p1, dp0, q_neumann, k, rule;
            identity_p1_p1=identity_p1_p1, identity_p1_dp0=identity_p1_dp0,
            skip_singular=false, singular_order=2, element_indices=element_indices,
            singular_cache=singular_cache, cpu_cache=cpu_cache, symmetry_mode=symmetry_mode,
        )

        @test fused.drive_count == drive_count
        @test size(fused.matrix) == (p1.global_dof_count, p1.global_dof_count)
        @test size(fused.rhs) == (p1.global_dof_count, drive_count)
        lhs_scale = max(norm(reference_lhs), eps(Float32))
        rhs_reference = reference_rhs_operator * q_neumann
        rhs_scale = max(norm(rhs_reference), eps(Float32))
        @test norm(fused.matrix - reference_lhs) / lhs_scale < 1.0f-5
        @test norm(fused.rhs - rhs_reference) / rhs_scale < 1.0f-5

        pressure = solve_burton_miller_neumann_system_cpu(fused)
        reference_pressure = lu(copy(reference_lhs)) \ rhs_reference
        @test norm(pressure - reference_pressure) / max(norm(reference_pressure), eps(Float32)) < 1.0f-4
    end
end

@testset "cpu x symmetry assembly" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_half.msh"), Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, :x)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        backend=:cpu,
        singular_cache=singular_cache,
        symmetry_mode=:x,
    )

    @test !get(operators, :on_gpu, true)
    @test operators.regular_pairs > length(element_indices) * length(element_indices)
    @test operators.singular_pairs == singular_cache.pair_count
    @test operators.image_singular_pairs >= 0
    @test sum(abs2, operators.single_layer) > 0
    @test all(isfinite, real.(operators.double_layer))
    @test all(isfinite, imag.(operators.double_layer))

    if cuda_available()
        cuda_cache = build_cuda_regular_assembly_cache(mesh, rule; element_indices=element_indices)
        cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)
        cuda_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=2,
            element_indices=element_indices,
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
            symmetry_mode=:x,
        )

        @test operators.single_layer ≈ Array(cuda_operators.single_layer) rtol=Float32(5e-3) atol=Float32(5e-5)
        @test operators.double_layer ≈ Array(cuda_operators.double_layer) rtol=Float32(5e-3) atol=Float32(5e-5)
        @test operators.adjoint_double_layer ≈ Array(cuda_operators.adjoint_double_layer) rtol=Float32(5e-3) atol=Float32(5e-5)
        @test operators.hypersingular ≈ Array(cuda_operators.hypersingular) rtol=Float32(5e-3) atol=Float32(5e-3)
        release_operator_storage!(cuda_operators)
    end

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=:x)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=:x)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    q_neumann[1] = ComplexF32(0, 1)
    pressure = solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    @test length(pressure) == p1.global_dof_count
    @test all(isfinite, real.(pressure))
    @test all(isfinite, imag.(pressure))

    field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=:x)
    eval_points = fibonacci_sphere(8, Float32(2.0))
    field = evaluate_galerkin_field_cpu(eval_points, mesh, pressure, q_neumann, k, field_cache)
    @test length(field) == length(eval_points)
    @test all(isfinite, real.(field))
    @test all(isfinite, imag.(field))
end

@testset "cpu xy symmetry assembly" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_quarter.msh"), Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, :xy)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        backend=:cpu,
        singular_cache=singular_cache,
        symmetry_mode=:xy,
    )
    cpu_cache = build_beat_cpu_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=2,
        element_indices=element_indices,
        symmetry_mode=:xy,
    )
    cached_operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        backend=:cpu,
        singular_cache=singular_cache,
        cpu_cache=cpu_cache,
        symmetry_mode=:xy,
    )

    @test !get(operators, :on_gpu, true)
    @test operators.regular_pairs > 2 * length(element_indices) * length(element_indices)
    @test operators.singular_pairs == singular_cache.pair_count
    @test operators.image_singular_pairs >= 0
    @test sum(abs2, operators.single_layer) > 0
    @test all(isfinite, real.(operators.hypersingular))
    @test all(isfinite, imag.(operators.hypersingular))
    @test cached_operators.single_layer ≈ operators.single_layer
    @test cached_operators.double_layer ≈ operators.double_layer
    @test cached_operators.adjoint_double_layer ≈ operators.adjoint_double_layer
    @test cached_operators.hypersingular ≈ operators.hypersingular
end

@testset "adaptive dense solve" begin
    @testset "cost model routes on both dofs and drives" begin
        # The measured picture on an M1 Max: at 10,230 dofs GMRES wins for one
        # drive only, and at 20,422 it wins up to three. A router on size alone
        # would send a three-way at 10,230 to GMRES and make it 2.2x slower.
        # Measured on the ATH ladder, per-frequency solve seconds:
        #   5,107  1 drive : GMRES 0.297 vs LU 0.725   -> GMRES
        #   5,107  3 drives: GMRES 0.941 vs LU 0.784   -> LU
        #  10,230  1 drive : GMRES 1.632 vs LU 5.601   -> GMRES
        #  20,422  1 drive : GMRES 5.063 vs LU 49.30   -> GMRES
        #  20,422  4 drives: GMRES 18.32 vs LU 48.21   -> GMRES
        #
        # Every number above is a consequence of the shipped M1 Max constants,
        # so the overrides are cleared for the duration: these assertions are
        # about the model at its documented calibration, not about the machine
        # running the suite. Without this the suite fails on any correctly
        # recalibrated host -- `scripts/calibrate_dense_solve.jl` measures 241.9
        # GFLOP/s and a 1,789-dof crossover on a Ryzen 7 5825U, which is a right
        # answer that turns two assertions below red.
        overrides = (
            BeatEngineCore.BEAT_DENSE_LU_GFLOPS_ENV,
            BeatEngineCore.BEAT_DENSE_MATVEC_ENTRY_SECONDS_ENV,
            BeatEngineCore.BEAT_DENSE_MATVEC_DOF_SECONDS_ENV,
            BeatEngineCore.BEAT_DENSE_TRIANGULAR_GBPS_ENV,
            BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_ENV,
        )
        withenv((name => nothing for name in overrides)...) do
            @test beat_dense_solve_plan(1_974, 1).method === :lu
            @test beat_dense_solve_plan(5_107, 1).method === :gmres
            @test beat_dense_solve_plan(5_107, 3).method === :lu
            @test beat_dense_solve_plan(10_230, 1).method === :gmres
            @test beat_dense_solve_plan(20_422, 1).method === :gmres
            @test beat_dense_solve_plan(20_422, 4).method === :gmres
            @test beat_dense_solve_plan(20_422, 24).method === :lu

            # Independently measured between 2,000 and 5,000 dofs at one drive.
            @test beat_dense_solve_crossover_dofs(1) > 2_000
            @test beat_dense_solve_crossover_dofs(1) < 5_000
        end

        # Machine-independent: more drives always pushes the crossover up,
        # never down, whatever the constants say. This one is deliberately
        # outside the block above.
        crossovers = [beat_dense_solve_crossover_dofs(drives) for drives in 1:6]
        @test issorted(crossovers)
    end

    @testset "wall-clock ceiling stops a run the iteration budget cannot" begin
        # The iteration budget bounds matvecs, not time. Orthogonalization is
        # O(m^2 N) and overtakes the matvec inside the budget's own range --
        # measured at 10,230 dofs, 385 iterations cost 32.8 s against 11.1 s of
        # matvec -- so only a clock can bound the damage.
        n = 300
        eigenvalues = ComplexF32[ComplexF32(1.05) + cis(Float32(2pi * i / n)) for i in 1:n]
        matrix = Matrix{ComplexF32}(Diagonal(eigenvalues))
        for row in 1:n, column in (row + 1):n
            matrix[row, column] += ComplexF32(cos(0.7f0 * row), sin(0.3f0 * column)) *
                                   0.5f0 / sqrt(Float32(n))
        end
        rhs = ComplexF32[ComplexF32(sin(0.2f0 * row), cos(0.11f0 * row)) for row in 1:n]

        # An unreachable tolerance runs until something else stops it. That
        # count is a property of the operator, not a number to hard-code -- it
        # is read here so the assertions below compare against it.
        solve(; kwargs...) = beat_gmres!(zeros(ComplexF32, n), matrix, copy(rhs);
                                         max_iterations=250, tolerance=1e-30,
                                         restart=0, kwargs...)
        natural = solve()
        # An already-expired real monotonic deadline must stop before work.
        # Comparing durations from separate solves is not reliable under load.
        bounded = solve(deadline_ns=time_ns() - UInt64(1))
        @test natural.iterations > 20
        @test bounded.iterations < natural.iterations
        @test bounded.converged == false
        @test bounded.reason === :deadline

        # Zero and negative mean no limit, so the default path is untouched.
        for none in (0, -1.0)
            @test solve(deadline_seconds=none).iterations == natural.iterations
        end

        # The cycle's solution update must happen before the timeout breaks out,
        # not after: leaving early without it hands back the zero iterate and
        # throws away every iteration that was paid for. Drive that boundary
        # with a deterministic monotonic clock; a real short deadline may
        # correctly expire before the first iteration under preemption.
        ticks = Ref(0)
        test_clock() = (ticks[] += 1; ticks[] == 1 ? UInt64(0) : UInt64(1))
        x = zeros(ComplexF32, n)
        partial = beat_gmres!(x, matrix, copy(rhs); max_iterations=250,
                              tolerance=1e-30, restart=0, deadline_ns=1,
                              clock_ns=test_clock)
        @test partial.iterations == 1
        @test partial.reason === :deadline
        @test any(!iszero, x)
    end

    @testset "iteration budget bounds a misrouted GMRES" begin
        # The budget is one LU's worth of matvecs, so a GMRES that exhausts it
        # has provably lost to the direct solve on that component. It does not
        # bound orthogonalization wall time; the deadline above does. Without
        # either guard the cap is min(n, 1000): measured 3.85x at
        # 4,751 dofs and 6 kHz, where the true iteration count is 429 against
        # the model's assumed 70.
        for (dofs, drives) in ((5_107, 1), (10_230, 1), (20_422, 4))
            budget = beat_gmres_iteration_budget(dofs, drives)
            spent = budget * beat_dense_matvec_seconds(dofs)
            @test spent <= beat_dense_lu_seconds(dofs, drives) * 1.01
        end

        # It has to leave room for the solves the model expects to win, or the
        # router would choose GMRES and then forbid it from finishing.
        for (dofs, drives) in ((5_107, 1), (10_230, 1), (20_422, 1))
            plan = beat_dense_solve_plan(dofs, drives)
            plan.method === :gmres || continue
            @test beat_gmres_iteration_budget(dofs, drives) >
                  BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_DEFAULT
        end

        # The budget is a total across drives, so it rises with drive count --
        # but only by the triangular solves, never by the factorization, which
        # is shared. The share each drive gets therefore falls, which is the
        # same asymmetry the router weighs.
        totals = [beat_gmres_iteration_budget(10_230, drives) for drives in 1:6]
        @test issorted(totals)
        @test issorted([total / drives for (drives, total) in enumerate(totals)]; rev=true)
    end

    @testset "explicit override beats the model" begin
        forced_lu = beat_dense_solve_plan(20_422, 1; method=:lu)
        @test forced_lu.method === :lu
        @test forced_lu.reason === :override
        forced_gmres = beat_dense_solve_plan(1_974, 1; method=:gmres)
        @test forced_gmres.method === :gmres
        @test forced_gmres.reason === :override
        @test beat_dense_solve_plan(1_974, 1).reason === :model
    end

    @testset "environment parsing" begin
        @test beat_dense_solve_method("auto") === :auto
        @test beat_dense_solve_method("") === :auto
        @test beat_dense_solve_method("LU") === :lu
        @test beat_dense_solve_method(" gmres ") === :gmres
        @test_throws ErrorException beat_dense_solve_method("magic")
    end

    @testset "the default gmres tolerance is 1e-5 and stays overridable" begin
        # 1e-6 floors above the target on the sliver-rim meshes and sends them to
        # the dense LU, paying a full GMRES budget and the factorization both.
        # 1e-5 converges there and moves the radiated field by 0.001 dB.
        @test BeatEngineCore._beat_gmres_tolerance(Float64) == 1.0e-5
        @test BeatEngineCore._beat_gmres_tolerance(Float32) == 1.0f-5

        withenv("BLAB_BEAT_GMRES_TOL" => "1e-6") do
            @test BeatEngineCore._beat_gmres_tolerance(Float64) == 1.0e-6
        end
        @test BeatEngineCore._beat_gmres_tolerance(Float64) == 1.0e-5
    end

    @testset "gmres and lu agree on the same system" begin
        n = 240
        rng_matrix = ComplexF32[
            ComplexF32(cos(0.31f0 * row + 0.17f0 * column), sin(0.23f0 * row - 0.41f0 * column)) / Float32(n)
            for row in 1:n, column in 1:n
        ]
        # Diagonally dominant enough to be well conditioned, as the assembled
        # Burton-Miller operator is once the identity block is added.
        matrix = rng_matrix + Matrix{ComplexF32}(2.0f0 * I, n, n)
        rhs = ComplexF32[
            ComplexF32(sin(0.7f0 * row + 1.1f0 * drive), cos(0.4f0 * row - 0.3f0 * drive))
            for row in 1:n, drive in 1:2
        ]

        reference = lu(copy(matrix)) \ rhs
        gmres_solution, gmres_report = beat_solve_dense_system(matrix, rhs; method=:gmres)
        @test gmres_report.method === :gmres
        @test !gmres_report.fell_back
        @test gmres_report.fallback_reason === nothing
        @test length(gmres_report.iterations) == 2
        @test all(reason -> reason === :converged, gmres_report.termination_reasons)
        # The default tolerance, not a tighter number that happens to hold: this
        # asserts the contract the router promises, and asserting 1e-6 here would
        # silently re-pin the default that `_beat_gmres_tolerance` documents.
        @test all(<=(1.0f-5), gmres_report.relative_residuals)
        @test norm(gmres_solution - reference) / norm(reference) < 1.0f-4

        lu_solution, lu_report = beat_solve_dense_system(matrix, rhs; method=:lu)
        @test lu_report.method === :lu
        @test isempty(lu_report.iterations)
        @test norm(lu_solution - reference) / norm(reference) < 1.0f-6

        # The matrix must survive both routes: the fused Metal path hands over
        # a shared device buffer the caller still owns.
        @test matrix[1, 1] == rng_matrix[1, 1] + 2.0f0
    end

    @testset "gmres solves a single-vector right-hand side" begin
        n = 96
        matrix = ComplexF32[
            ComplexF32(cos(0.5f0 * row * column), sin(0.25f0 * (row + column))) / Float32(n)
            for row in 1:n, column in 1:n
        ] + Matrix{ComplexF32}(3.0f0 * I, n, n)
        rhs = ComplexF32[ComplexF32(row / n, -row / (2n)) for row in 1:n]
        x = zeros(ComplexF32, n)
        result = beat_gmres!(x, matrix, rhs)
        @test result.converged
        @test result.relative_residual <= 1.0f-6
        @test norm(matrix * x - rhs) / norm(rhs) <= 1.0f-6
    end

    @testset "non-convergence falls back to the lu instead of failing" begin
        # A deliberately hostile system with a one-iteration budget: GMRES
        # cannot converge, and the caller must still get the right answer.
        n = 64
        matrix = ComplexF32[
            ComplexF32(cos(3.1f0 * row * column), sin(2.7f0 * row - 1.3f0 * column))
            for row in 1:n, column in 1:n
        ] + Matrix{ComplexF32}(0.05f0 * I, n, n)
        rhs = ComplexF32[ComplexF32(sin(0.9f0 * row), cos(0.6f0 * row)) for row in 1:n]
        reference = lu(copy(matrix)) \ rhs

        withenv("BLAB_BEAT_GMRES_MAX_ITERATIONS" => "1") do
            solution, report = beat_solve_dense_system(matrix, reshape(rhs, :, 1); method=:gmres)
            @test report.fell_back
            @test report.method === :lu
            @test report.fallback_reason === :iteration_limit
            @test report.termination_reasons == [:iteration_limit]
            @test occursin("per-drive iteration limit", describe_dense_solve(report))
            @test norm(vec(solution) - reference) / norm(reference) < 1.0f-4
        end

        # Auto routing applies both budgets and says which one stopped it. A
        # forced GMRES remains forced even when the model deadline is tiny.
        withenv("BLAB_BEAT_GMRES_MODEL_ITERATIONS" => "1e-9",
                "BLAB_BEAT_GMRES_BUDGET" => "1e-12",
                "BLAB_BEAT_GMRES_TIME_CEILING" => "1e12") do
            solution, report = beat_solve_dense_system(matrix, reshape(rhs, :, 1))
            @test report.plan.reason === :model
            @test report.fell_back
            @test report.fallback_reason === :iteration_budget
            @test report.termination_reasons == [:iteration_limit]
            @test occursin("shared iteration budget", describe_dense_solve(report))
            @test norm(vec(solution) - reference) / norm(reference) < 1.0f-4
        end

        withenv("BLAB_BEAT_GMRES_MODEL_ITERATIONS" => "1e-9",
                "BLAB_BEAT_GMRES_BUDGET" => "1e12",
                "BLAB_BEAT_GMRES_TIME_CEILING" => "1e-12") do
            solution, report = beat_solve_dense_system(matrix, hcat(rhs, rhs))
            @test report.plan.reason === :model
            @test report.plan.drives == 2
            @test report.fell_back
            @test report.fallback_reason === :deadline
            @test occursin("shared wall-clock deadline", describe_dense_solve(report))
            @test norm(solution[:, 1] - reference) / norm(reference) < 1.0f-4

            benign = Matrix{ComplexF32}(2.0f0 * I, n, n)
            forced_solution, forced_report = beat_solve_dense_system(
                benign, reshape(rhs, :, 1); method=:gmres,
            )
            @test !forced_report.fell_back
            @test forced_report.method === :gmres
            @test norm(benign * forced_solution[:, 1] - rhs) / norm(rhs) < 1.0f-5
        end
    end

    @testset "zero right-hand side" begin
        n = 32
        matrix = Matrix{ComplexF32}(2.0f0 * I, n, n)
        x = ones(ComplexF32, n)
        result = beat_gmres!(x, matrix, zeros(ComplexF32, n))
        @test result.converged
        @test result.reason === :converged
        @test all(iszero, x)
    end


    @testset "krylov space precision and orthogonality" begin
        # A spectrum on a circle that nearly touches the origin: GMRES has to
        # build a real Krylov space rather than terminating in a few steps.
        # The BEAT operator's own failure appeared past 200 iterations, which
        # no test here reaches -- see scripts/validate_gmres_burton_miller.jl
        # for the guard that runs on a real operator. What these assert are the
        # invariants that were violated, which hold at any length.
        n = 400
        eigenvalues = ComplexF32[ComplexF32(1.1) + cis(Float32(2pi * index / n)) for index in 1:n]
        matrix = Matrix{ComplexF32}(Diagonal(eigenvalues))
        for row in 1:n, column in (row + 1):n
            matrix[row, column] += ComplexF32(cos(0.9f0 * row + 0.4f0 * column),
                                              sin(0.6f0 * row - 0.8f0 * column)) * 0.5f0 / sqrt(Float32(n))
        end
        rhs = ComplexF32[ComplexF32(sin(0.31f0 * row), cos(0.17f0 * row)) for row in 1:n]

        # Pinned, not inherited. This testset exists to catch orthogonality loss,
        # and orthogonality loss only shows up in a long run -- so the tolerance
        # has to be tight enough to keep the run long, independently of whatever
        # the production default is. Taking the default here would have quietly
        # shortened the case when that default loosened to 1e-5.
        function run(krylov_type, reorthogonalize; restart=0)
            x = zeros(ComplexF32, n)
            result = beat_gmres!(x, matrix, copy(rhs); krylov_type=krylov_type,
                                 reorthogonalize=reorthogonalize, restart=restart,
                                 tolerance=1.0e-6, max_iterations=2000)
            return result, x
        end

        float64_result, float64_x = run(ComplexF64, :dgks)
        @test float64_result.converged
        # The case must be long enough to be worth running. A test that
        # converges in five iterations cannot catch an orthogonality failure,
        # which is exactly how the original bug survived its own unit tests.
        @test float64_result.iterations >= 15
        # Independently recomputed, so it differs from the solver's own fused
        # residual by Float32 rounding. The bound is above 1e-6 for that
        # reason, not because the solve is loose: at N=400 the achievable true
        # residual in Float32 is only a few times sqrt(N)*eps.
        @test norm(matrix * float64_x - rhs) / norm(rhs) <= 3.0f-6

        # Two independent remedies must agree with each other. Agreement is the
        # evidence; neither count alone is.
        reorthogonalized_result, reorthogonalized_x = run(ComplexF32, :always)
        @test reorthogonalized_result.converged
        @test abs(reorthogonalized_result.iterations - float64_result.iterations) <= 2
        @test norm(reorthogonalized_x - float64_x) / norm(float64_x) < 1.0f-3

        # The failure the remedies protect against should be reachable here, or
        # the agreement assertions above are weak. But whether a Float32
        # recurrence loses orthogonality on a given system is a property of the
        # *host's* floating point, not of the code under test: this system
        # degrades 20x on an M1 Max and reportedly not at all on a Ryzen 7
        # 5825U. Failing there would report a microarchitecture, not a defect.
        #
        # So this warns rather than asserts. The agreement between the three
        # remedies stays a hard assertion; what is conditional is only how much
        # that agreement proves on this particular machine. The hard version of
        # this guard lives in scripts/validate_gmres_burton_miller.jl, against a
        # real operator, where the margin is 1000 iterations against 51.
        stalled_result, _ = run(ComplexF32, :never)
        if stalled_result.iterations <= 4 * float64_result.iterations
            @warn "Unreorthogonalized Float32 MGS did not degrade on this host, so " *
                  "the Krylov agreement assertions above are weaker here than intended." *
                  " single MGS: $(stalled_result.iterations) iterations, " *
                  "Float64: $(float64_result.iterations)."
        end
        @test stalled_result.iterations >= float64_result.iterations

        # A converging solver does not care what the restart is, as long as the
        # restart exceeds the count it converges in. An iteration count that
        # tracks the restart parameter is not an iteration count -- that was
        # the tell that identified the Float32 Arnoldi failure.
        for restart in (float64_result.iterations + 20, float64_result.iterations + 100, 0)
            restarted, _ = run(ComplexF64, :dgks; restart=restart)
            @test restarted.iterations == float64_result.iterations
        end
    end

    @testset "krylov precision and reorthogonalization parsing" begin
        @test beat_gmres_krylov_type("f64") === ComplexF64
        @test beat_gmres_krylov_type("") === ComplexF64
        @test beat_gmres_krylov_type("F32") === ComplexF32
        @test_throws ErrorException beat_gmres_krylov_type("f16")
        @test beat_gmres_reorthogonalization("dgks") === :dgks
        @test beat_gmres_reorthogonalization("always") === :always
        @test beat_gmres_reorthogonalization("never") === :never
        @test_throws ErrorException beat_gmres_reorthogonalization("sometimes")
    end

    @testset "an unreachable tolerance stalls out instead of burning the budget" begin
        # 1e-12 is below the Float32 residual floor, so no number of cycles
        # reaches it. The run must report the stall promptly and let the caller
        # fall back, not spend its whole allowance discovering that. It must
        # also not exceed the Krylov dimension per cycle.
        n = 40
        matrix = ComplexF32[
            ComplexF32(cos(0.5f0 * row * column), sin(0.25f0 * (row + column))) / Float32(n)
            for row in 1:n, column in 1:n
        ] + Matrix{ComplexF32}(1.5f0 * I, n, n)
        rhs = ComplexF32[ComplexF32(row / n, -row / (2n)) for row in 1:n]
        x = zeros(ComplexF32, n)
        result = beat_gmres!(x, matrix, rhs; max_iterations=5000, tolerance=1.0e-12)
        @test !result.converged
        @test result.iterations <= 2n
        # It still solved the system as well as Float32 permits.
        @test norm(matrix * x - rhs) / norm(rhs) < 1.0f-5
    end

    @testset "diagonal preconditioner tolerates a zero diagonal entry" begin
        matrix = ComplexF32[1 2; 3 0]
        inverse = beat_diagonal_preconditioner(matrix)
        @test inverse[1] == ComplexF32(1)
        @test inverse[2] == ComplexF32(1)
        @test all(isfinite, inverse)
    end
end

@testset "rigid y0 half-space Green function" begin
    T = Float32
    direct_vertices = [
        SVector{3,T}(0, 0.2, 0),
        SVector{3,T}(0.04, 0.2, 0),
        SVector{3,T}(0, 0.2, 0.04),
    ]
    direct_mesh = BoundaryMesh(direct_vertices, [(1, 2, 3)], [1])
    full_vertices = vcat(
        direct_vertices,
        [SVector{3,T}(point[1], -point[2], point[3]) for point in direct_vertices],
    )
    # Reverse the reflected triangle so its outward normal is the physical
    # reflection of the direct triangle's normal.
    full_mesh = BoundaryMesh(full_vertices, [(1, 2, 3), (4, 6, 5)], [1, 1])
    direct_p1 = build_p1_space(direct_mesh)
    direct_dp0 = build_dp0_space(direct_mesh)
    full_p1 = build_p1_space(full_mesh)
    full_dp0 = build_dp0_space(full_mesh)
    rule = triangle_rule(T, 3)
    k = T(2pi * 80.0 / 343.0)

    direct_singular = build_singular_correction_cache(direct_mesh, 3)
    full_singular = build_singular_correction_cache(full_mesh, 3)
    half_space = assemble_regular_galerkin_operators(
        direct_mesh, direct_p1, direct_dp0, k, rule;
        backend=:cpu, skip_singular=false, singular_cache=direct_singular,
        symmetry_mode=:ground,
    )
    full_space = assemble_regular_galerkin_operators(
        full_mesh, full_p1, full_dp0, k, rule;
        backend=:cpu, skip_singular=false, singular_cache=full_singular,
    )

    @test rigid_ground_transform().signs == SVector{3,Int}(1, -1, 1)
    @test half_space.single_layer[:, 1] ≈
        full_space.single_layer[1:3, 1] + full_space.single_layer[1:3, 2] rtol=T(2e-5)
    @test half_space.adjoint_double_layer[:, 1] ≈
        full_space.adjoint_double_layer[1:3, 1] + full_space.adjoint_double_layer[1:3, 2] rtol=T(2e-5) atol=T(2e-7)
    @test half_space.double_layer ≈
        full_space.double_layer[1:3, 1:3] + full_space.double_layer[1:3, 4:6] rtol=T(2e-5) atol=T(2e-7)
    @test half_space.hypersingular ≈
        full_space.hypersingular[1:3, 1:3] + full_space.hypersingular[1:3, 4:6] rtol=T(2e-5) atol=T(2e-5)

    identity_off = assemble_l2_identity_matrix(direct_mesh, direct_p1, direct_dp0, rule, :p1, :p1)
    identity_ground = assemble_l2_identity_matrix(
        direct_mesh, direct_p1, direct_dp0, rule, :p1, :p1;
        symmetry_mode=:ground,
    )
    @test identity_ground == identity_off

    pressure = Complex{T}[1.0 + 0.2im, 0.7 - 0.1im, 1.2 + 0.05im]
    neumann = Complex{T}[0.3 - 0.2im]
    field_cache = build_field_evaluation_cache(direct_mesh, rule; symmetry_mode=:ground)
    mirrored_field = evaluate_galerkin_field_cpu(
        [SVector{3,T}(0.01, 0.6, 0.02), SVector{3,T}(0.01, -0.6, 0.02)],
        direct_mesh,
        pressure,
        neumann,
        k,
        field_cache,
    )
    @test mirrored_field[1] ≈ mirrored_field[2] rtol=T(2e-6) atol=T(2e-7)

    if cuda_available()
        cuda_regular = build_cuda_regular_assembly_cache(direct_mesh, rule)
        cuda_singular = BeatEngineCore.build_cuda_singular_correction_cache(
            direct_singular,
            direct_p1,
            direct_dp0,
        )
        cuda_image_singular = build_cuda_image_singular_correction_cache(
            direct_mesh,
            direct_p1,
            direct_dp0,
            3,
            eachindex(direct_mesh.faces),
            :ground,
        )
        cuda_half_space = assemble_regular_galerkin_operators(
            direct_mesh, direct_p1, direct_dp0, k, rule;
            skip_singular=false,
            device_cache=cuda_regular,
            singular_cache=direct_singular,
            device_singular_cache=cuda_singular,
            device_image_singular_cache=cuda_image_singular,
            symmetry_mode=:ground,
        )
        @test Array(cuda_half_space.single_layer) ≈ half_space.single_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_half_space.double_layer) ≈ half_space.double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_half_space.adjoint_double_layer) ≈ half_space.adjoint_double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_half_space.hypersingular) ≈ half_space.hypersingular rtol=T(5e-3) atol=T(5e-3)

        cuda_field_cache = build_cuda_field_evaluation_cache(field_cache)
        cuda_field = evaluate_galerkin_field_cuda(
            [SVector{3,T}(0.01, 0.6, 0.02), SVector{3,T}(0.01, -0.6, 0.02)],
            direct_mesh,
            pressure,
            neumann,
            k,
            cuda_field_cache,
        )
        @test cuda_field ≈ mirrored_field rtol=T(5e-4) atol=T(5e-6)
        release_operator_storage!(cuda_half_space)
        release_cuda_image_singular_correction_cache!(cuda_image_singular)
    end
end

@testset "close-pair higher-order correction" begin
    T = Float32
    vertices = [
        SVector{3,T}(0, 0, 0),
        SVector{3,T}(0.04, 0, 0),
        SVector{3,T}(0, 0.04, 0),
        SVector{3,T}(0, 0, 0.01),
        SVector{3,T}(0.04, 0, 0.01),
        SVector{3,T}(0, 0.04, 0.01),
    ]
    mesh = BoundaryMesh(vertices, [(1, 2, 3), (4, 5, 6)], [1, 1])
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    base_rule = triangle_rule(T, 2)
    near_cache = build_near_correction_cache(mesh, [(1, 2, 4), (2, 1, 6)], 6)
    empty_near_cache = build_near_correction_cache(mesh, Tuple{Int,Int}[], 6)
    singular_cache = build_singular_correction_cache(mesh, 2)
    k = T(2pi * 100.0 / 343.0)

    @test near_cache.pair_count == 2
    @test empty_near_cache.pair_count == 0
    @test empty_near_cache.correction_orders == Int[]
    @test near_cache.correction_order == 6
    @test near_cache.correction_orders == [4, 6]
    @test length(near_cache.correction_rules[1].points) == 16
    @test length(near_cache.correction_rules[2].points) == 36
    @test sum(near_cache.correction_rules[1].weights) ≈ T(0.5) atol=T(1e-6)
    @test sum(near_cache.correction_rules[2].weights) ≈ T(0.5) atol=T(1e-6)

    base = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
    )
    corrected = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
        near_correction_cache=near_cache,
    )
    reference_forward = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, near_cache.correction_rules[1];
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
    )
    reference_reverse = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, near_cache.correction_rules[2];
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
    )

    @test corrected.near_pair_count == 2
    @test corrected.near_pair_quadrature_order == 6
    @test corrected.single_layer[1:3, 2] ≈ reference_forward.single_layer[1:3, 2] rtol=T(2e-5)
    @test corrected.double_layer[1:3, 4:6] ≈ reference_forward.double_layer[1:3, 4:6] rtol=T(2e-5) atol=T(2e-7)
    @test corrected.adjoint_double_layer[1:3, 2] ≈ reference_forward.adjoint_double_layer[1:3, 2] rtol=T(2e-5) atol=T(2e-7)
    @test corrected.hypersingular[1:3, 4:6] ≈ reference_forward.hypersingular[1:3, 4:6] rtol=T(2e-5) atol=T(2e-5)
    @test corrected.single_layer[4:6, 1] ≈ reference_reverse.single_layer[4:6, 1] rtol=T(2e-5)
    @test norm(corrected.single_layer[1:3, 2] - base.single_layer[1:3, 2]) > T(1e-9)

    ground_image = SymmetryTransform(:ground_image, SVector{3,Int}(1, -1, 1), -1)
    image_cache = build_near_correction_cache(
        mesh,
        [(1, 2)],
        6;
        trial_transform=ground_image,
    )
    @test image_cache.pair_count == 1
    @test image_cache.trial_transform == ground_image
    combined_image_corrected = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
        near_correction_cache=near_cache,
        image_near_correction_cache=image_cache,
        symmetry_mode=:ground,
    )
    @test combined_image_corrected.near_pair_count == 3
    @test all(isfinite, real.(combined_image_corrected.single_layer))

    image_mesh = BoundaryMesh(
        [
            SVector{3,T}(0.01, 0, 0),
            SVector{3,T}(0.01, 0.04, 0),
            SVector{3,T}(0.01, 0, 0.04),
        ],
        [(1, 2, 3)],
        [1],
    )
    image_p1 = build_p1_space(image_mesh)
    image_dp0 = build_dp0_space(image_mesh)
    x_image = SymmetryTransform(:reflect_x, SVector{3,Int}(-1, 1, 1), -1)
    reflected_near_cache = build_near_correction_cache(
        image_mesh,
        [(1, 1)],
        6;
        trial_transform=x_image,
    )
    image_singular_cache = build_singular_correction_cache(image_mesh, 2)
    reflected_corrected = assemble_regular_galerkin_operators(
        image_mesh, image_p1, image_dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=image_singular_cache,
        near_correction_cache=reflected_near_cache, symmetry_mode=:x,
    )
    reflected_reference = assemble_regular_galerkin_operators(
        image_mesh, image_p1, image_dp0, k, reflected_near_cache.correction_rules[1];
        backend=:cpu, skip_singular=false, singular_cache=image_singular_cache,
        symmetry_mode=:x,
    )
    @test reflected_corrected.single_layer ≈ reflected_reference.single_layer rtol=T(2e-5)
    @test reflected_corrected.double_layer ≈ reflected_reference.double_layer rtol=T(2e-5) atol=T(2e-7)
    @test reflected_corrected.adjoint_double_layer ≈ reflected_reference.adjoint_double_layer rtol=T(2e-5) atol=T(2e-7)
    @test reflected_corrected.hypersingular ≈ reflected_reference.hypersingular rtol=T(2e-5) atol=T(2e-5)

    if cuda_available()
        cuda_regular = build_cuda_regular_assembly_cache(mesh, base_rule)
        cuda_singular = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)
        cuda_near = build_cuda_near_correction_cache(near_cache, p1, dp0)
        cuda_corrected = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, base_rule;
            skip_singular=false,
            device_cache=cuda_regular,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular,
            near_correction_cache=near_cache,
            device_near_correction_cache=cuda_near,
        )
        @test cuda_corrected.near_pair_count == near_cache.pair_count
        @test Array(cuda_corrected.single_layer) ≈ corrected.single_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_corrected.double_layer) ≈ corrected.double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_corrected.adjoint_double_layer) ≈ corrected.adjoint_double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_corrected.hypersingular) ≈ corrected.hypersingular rtol=T(5e-3) atol=T(5e-3)
        cuda_image_near = build_cuda_near_correction_cache(image_cache, p1, dp0)
        cuda_image_singular = build_cuda_image_singular_correction_cache(
            mesh, p1, dp0, 2, eachindex(mesh.faces), :ground,
        )
        cuda_combined = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, base_rule;
            skip_singular=false,
            device_cache=cuda_regular,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular,
            device_image_singular_cache=cuda_image_singular,
            near_correction_cache=near_cache,
            device_near_correction_cache=cuda_near,
            image_near_correction_cache=image_cache,
            device_image_near_correction_cache=cuda_image_near,
            symmetry_mode=:ground,
        )
        @test cuda_combined.near_pair_count == combined_image_corrected.near_pair_count
        @test Array(cuda_combined.single_layer) ≈ combined_image_corrected.single_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_combined.double_layer) ≈ combined_image_corrected.double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_combined.adjoint_double_layer) ≈ combined_image_corrected.adjoint_double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_combined.hypersingular) ≈ combined_image_corrected.hypersingular rtol=T(5e-3) atol=T(5e-3)

        ground_identity_p1_p1 = assemble_l2_identity_matrix(
            mesh, p1, dp0, base_rule, :p1, :p1; symmetry_mode=:ground,
        )
        ground_identity_p1_dp0 = assemble_l2_identity_matrix(
            mesh, p1, dp0, base_rule, :p1, :dp0; symmetry_mode=:ground,
        )
        ground_identity_cache = build_cuda_burton_miller_identity_cache(
            ground_identity_p1_p1, ground_identity_p1_dp0, T,
        )
        direct_q = Complex{T}[Complex{T}(0, 1), Complex{T}(0.25, -0.5)]
        d_direct_q = CUDA_MODULE.CuArray(direct_q)
        direct_coupling = Complex{T}(0, 1) / k
        expected_direct_lhs = (
            Complex{T}(0.5) .* ground_identity_cache.identity_p1_p1 .-
            cuda_combined.double_layer .+
            direct_coupling .* cuda_combined.hypersingular
        )
        expected_direct_rhs = (
            -cuda_combined.single_layer .-
            direct_coupling .* (
                cuda_combined.adjoint_double_layer .+
                Complex{T}(0.5) .* ground_identity_cache.identity_p1_dp0
            )
        ) * d_direct_q
        corrected_direct_system = assemble_burton_miller_neumann_system_cuda(
            mesh,
            p1,
            dp0,
            d_direct_q,
            k,
            base_rule;
            device_cache=cuda_regular,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular,
            device_image_singular_cache=cuda_image_singular,
            near_correction_cache=near_cache,
            device_near_correction_cache=cuda_near,
            image_near_correction_cache=image_cache,
            device_image_near_correction_cache=cuda_image_near,
            symmetry_mode=:ground,
        )
        @test corrected_direct_system.near_pair_count == combined_image_corrected.near_pair_count
        @test Array(corrected_direct_system.matrix) ≈ Array(expected_direct_lhs) rtol=T(5e-3) atol=T(5e-4)
        @test Array(corrected_direct_system.rhs) ≈ Array(expected_direct_rhs) rtol=T(5e-3) atol=T(5e-4)
        release_burton_miller_system_cuda!(corrected_direct_system)
        CUDA_MODULE.unsafe_free!(expected_direct_lhs)
        CUDA_MODULE.unsafe_free!(expected_direct_rhs)
        CUDA_MODULE.unsafe_free!(d_direct_q)
        release_cuda_burton_miller_identity_cache!(ground_identity_cache)
        release_operator_storage!(cuda_combined)
        release_cuda_image_singular_correction_cache!(cuda_image_near)
        release_cuda_image_singular_correction_cache!(cuda_image_singular)
        release_operator_storage!(cuda_corrected)
        release_cuda_image_singular_correction_cache!(cuda_near)
    end
end

@testset "cuda production pipeline" begin
    if !cuda_available()
        @test_skip "CUDA unavailable; skipping CUDA-only BEAT Engine tests."
    else
        mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(Float32, 2)
        k = Float32(2pi * 1000.0 / 343.0)
        element_indices = 1:min(16, length(mesh.faces))
        singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
        cuda_cache = build_cuda_regular_assembly_cache(mesh, rule; element_indices=element_indices)
        cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)

        operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=2,
            element_indices=element_indices,
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
        )

        @test get(operators, :on_gpu, false)
        @test operators.regular_assembly_mode == :serial_pair_batched
        @test operators.regular_kernel_mode == "serial_pair_batched"
        @test operators.regular_pairs > 0
        @test operators.singular_pairs == singular_cache.pair_count
        @test BeatEngineCore._cuda_use_matrix_free_burton_miller_rhs(operators)

        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
        identity_cache = build_cuda_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
        q_neumann = zeros(ComplexF32, length(mesh.faces))
        q_neumann[1] = ComplexF32(0, 1)
        d_q_neumann = CUDA_MODULE.CuArray(q_neumann)
        coupling = ComplexF32(0, 1) / k
        d_expected_rhs = (
            -operators.single_layer .-
            coupling .* (operators.adjoint_double_layer .+ ComplexF32(0.5) .* identity_cache.identity_p1_dp0)
        ) * d_q_neumann
        d_expected_lhs = (
            ComplexF32(0.5) .* identity_cache.identity_p1_p1 .-
            operators.double_layer .+
            coupling .* operators.hypersingular
        )
        d_matrix_free_rhs = BeatEngineCore._cuda_burton_miller_rhs(operators, identity_cache, d_q_neumann, coupling)
        direct_timing = Dict{String,Float64}()
        direct_system = assemble_burton_miller_neumann_system_cuda(
            mesh,
            p1,
            dp0,
            d_q_neumann,
            k,
            rule;
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
            timing=direct_timing,
        )
        @test direct_system.assembly_mode == :direct_burton_miller
        @test Array(direct_system.matrix) ≈ Array(d_expected_lhs) rtol=2f-5 atol=2f-6
        @test Array(direct_system.rhs) ≈ Array(d_expected_rhs) rtol=2f-5 atol=2f-6
        @test haskey(direct_timing, "direct_system_regular")
        d_direct_rhs_only = assemble_burton_miller_rhs_cuda(
            mesh,
            p1,
            dp0,
            d_q_neumann,
            k,
            rule;
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
        )
        @test Array(d_direct_rhs_only) ≈ Array(d_expected_rhs) rtol=2f-5 atol=2f-6
        CUDA_MODULE.unsafe_free!(d_direct_rhs_only)
        direct_pressure = solve_burton_miller_system_cuda!(direct_system)
        @test Array(d_matrix_free_rhs) ≈ Array(d_expected_rhs) rtol=2f-5
        CUDA_MODULE.unsafe_free!(d_q_neumann)
        CUDA_MODULE.unsafe_free!(d_expected_lhs)
        CUDA_MODULE.unsafe_free!(d_expected_rhs)
        CUDA_MODULE.unsafe_free!(d_matrix_free_rhs)

        pressure = solve_burton_miller_neumann(operators, identity_cache, q_neumann, k)
        release_cuda_burton_miller_identity_cache!(identity_cache)

        @test length(pressure) == p1.global_dof_count
        @test direct_pressure ≈ pressure rtol=2f-4 atol=2f-5
        @test all(isfinite, real.(pressure))
        @test all(isfinite, imag.(pressure))

        field_cache = build_cuda_field_evaluation_cache(build_field_evaluation_cache(mesh, rule))
        eval_points = fibonacci_sphere(8, Float32(2.0))
        field = evaluate_galerkin_field_cuda(eval_points, mesh, pressure, q_neumann, k, field_cache)
        @test length(field) == length(eval_points)
        @test all(isfinite, real.(field))
        @test all(isfinite, imag.(field))

        release_operator_storage!(operators)

        symmetry_mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_quarter.msh"), Float32(0.001))
        symmetry_p1 = build_p1_space(symmetry_mesh)
        symmetry_dp0 = build_dp0_space(symmetry_mesh)
        symmetry_indices = eachindex(symmetry_mesh.faces)
        symmetry_singular_cache = build_singular_correction_cache(symmetry_mesh, 2, symmetry_indices)
        symmetry_cuda_cache = build_cuda_regular_assembly_cache(symmetry_mesh, rule; element_indices=symmetry_indices)
        symmetry_cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(
            symmetry_singular_cache,
            symmetry_p1,
            symmetry_dp0,
        )
        image_cache = build_cuda_image_singular_correction_cache(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            2,
            symmetry_indices,
            :xy,
        )
        image_timing = Dict{String,Float64}()
        symmetry_operators = assemble_regular_galerkin_operators(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=2,
            element_indices=symmetry_indices,
            device_cache=symmetry_cuda_cache,
            singular_cache=symmetry_singular_cache,
            device_singular_cache=symmetry_cuda_singular_cache,
            device_image_singular_cache=image_cache,
            symmetry_mode=:xy,
            timing=image_timing,
        )
        @test image_cache.pair_count > 0
        @test symmetry_operators.image_singular_pairs == image_cache.pair_count
        @test image_timing["image_singular_correction_cuda_cache_build"] == 0.0
        @test !BeatEngineCore._cuda_use_matrix_free_burton_miller_rhs(symmetry_operators)

        symmetry_identity_p1_p1 = assemble_l2_identity_matrix(
            symmetry_mesh, symmetry_p1, symmetry_dp0, rule, :p1, :p1; symmetry_mode=:xy,
        )
        symmetry_identity_p1_dp0 = assemble_l2_identity_matrix(
            symmetry_mesh, symmetry_p1, symmetry_dp0, rule, :p1, :dp0; symmetry_mode=:xy,
        )
        symmetry_identity_cache = build_cuda_burton_miller_identity_cache(
            symmetry_identity_p1_p1, symmetry_identity_p1_dp0, Float32,
        )
        symmetry_q = zeros(ComplexF32, symmetry_dp0.global_dof_count)
        symmetry_q[1] = ComplexF32(0, 1)
        d_symmetry_q = CUDA_MODULE.CuArray(symmetry_q)
        symmetry_expected_lhs = (
            ComplexF32(0.5) .* symmetry_identity_cache.identity_p1_p1 .-
            symmetry_operators.double_layer .+
            coupling .* symmetry_operators.hypersingular
        )
        symmetry_expected_rhs = (
            -symmetry_operators.single_layer .-
            coupling .* (
                symmetry_operators.adjoint_double_layer .+
                ComplexF32(0.5) .* symmetry_identity_cache.identity_p1_dp0
            )
        ) * d_symmetry_q
        symmetry_direct_system = assemble_burton_miller_neumann_system_cuda(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            d_symmetry_q,
            k,
            rule;
            device_cache=symmetry_cuda_cache,
            singular_cache=symmetry_singular_cache,
            device_singular_cache=symmetry_cuda_singular_cache,
            device_image_singular_cache=image_cache,
            symmetry_mode=:xy,
        )
        @test symmetry_direct_system.image_singular_pairs == image_cache.pair_count
        @test Array(symmetry_direct_system.matrix) ≈ Array(symmetry_expected_lhs) rtol=5f-4 atol=5f-5
        @test Array(symmetry_direct_system.rhs) ≈ Array(symmetry_expected_rhs) rtol=5f-4 atol=5f-5
        symmetry_rhs_only = assemble_burton_miller_rhs_cuda(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            d_symmetry_q,
            k,
            rule;
            device_cache=symmetry_cuda_cache,
            singular_cache=symmetry_singular_cache,
            device_singular_cache=symmetry_cuda_singular_cache,
            device_image_singular_cache=image_cache,
            symmetry_mode=:xy,
        )
        @test Array(symmetry_rhs_only) ≈ Array(symmetry_expected_rhs) rtol=5f-4 atol=5f-5
        CUDA_MODULE.unsafe_free!(symmetry_rhs_only)
        release_burton_miller_system_cuda!(symmetry_direct_system)
        CUDA_MODULE.unsafe_free!(symmetry_expected_lhs)
        CUDA_MODULE.unsafe_free!(symmetry_expected_rhs)
        CUDA_MODULE.unsafe_free!(d_symmetry_q)
        release_cuda_burton_miller_identity_cache!(symmetry_identity_cache)
        release_operator_storage!(symmetry_operators)
        release_cuda_image_singular_correction_cache!(image_cache)
    end
end
