# Ordering and reproducibility gate for the sweep assembly pipeline.
#
# The sweep runs the Metal assembly on a producer task several frequencies
# ahead of the solve. Two things have to survive that, and neither shows up in
# a timing number:
#
#   * every step is delivered in ascending order carrying the operators of its
#     own frequency, because the consumer labels results positionally;
#   * running ahead introduces no divergence of its own. The fused path's
#     pair-block scratch lives in the shared assembly cache, so two concurrent
#     assemblies would race on it -- which is exactly why the producer is a
#     single task, and exactly what this script would catch if it stopped being
#     one.
#
# The second check needs care, because the *shipped* configuration is not
# bit-reproducible to begin with: the native singular scatter accumulates with
# atomics, so two sequential assemblies of the same frequency already differ by
# a few times 1e-7 relative. Comparing a pipelined run against a sequential one
# there can only ever show that floor. So the script runs two arms:
#
#   deterministic  four operators, host singular corrections. Sequential runs
#                  are bit-identical, so the pipelined run must be too. This is
#                  the arm that can actually detect a race.
#   shipped        the fused Burton-Miller path with native singular scatter.
#                  Pipelined-versus-sequential must be no worse than
#                  sequential-versus-sequential, measured in the same run.
#
#   BLAB_VALIDATE_MESH_PATH   absolute mesh path (default: the bundled sample)
#   BLAB_VALIDATE_MESH        bundled fixture name (default sample.msh)
#   BLAB_VALIDATE_SCALE       mesh scale (default 0.001 for the sample)
#   BLAB_VALIDATE_SYMMETRY    off | x | xy
#   BLAB_VALIDATE_DRIVES      number of independent drive columns (default 2)
#   BLAB_VALIDATE_STEPS       frequencies in the sweep (default 12)
using LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

relative_difference(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(Float32))

_normalized_metal_singular_mode_is_native() =
    BeatEngineCore._normalized_metal_singular_mode() == :native

function validate_metal_sweep_pipeline()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")
    Threads.nthreads() > 1 ||
        error("The sweep pipeline needs a second Julia thread; start julia with --threads.")

    mesh_path = get(ENV, "BLAB_VALIDATE_MESH_PATH", "")
    scale = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_SCALE", isempty(mesh_path) ? "0.001" : "1.0")))
    if isempty(mesh_path)
        mesh_path = joinpath(@__DIR__, "..", "test_meshes", get(ENV, "BLAB_VALIDATE_MESH", "sample.msh"))
    end
    symmetry_mode = Symbol(get(ENV, "BLAB_VALIDATE_SYMMETRY", "off"))
    drive_count = parse(Int, get(ENV, "BLAB_VALIDATE_DRIVES", "2"))
    regular_order = parse(Int, get(ENV, "BLAB_VALIDATE_REGULAR_ORDER", "4"))
    singular_order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "4"))
    steps = parse(Int, get(ENV, "BLAB_VALIDATE_STEPS", "12"))

    mesh = load_gmsh22_with_tags(mesh_path, scale)
    mesh = snap_symmetry_planes(mesh, symmetry_mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, regular_order)
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)

    frequencies = Float32.(exp10.(range(log10(100.0), log10(20000.0); length=steps)))
    wavenumbers = Float32(2pi) .* frequencies ./ 343.0f0

    singular_mode = BeatEngineCore._normalized_metal_singular_mode()
    println("fixture=$(mesh_path) scale=$(scale) symmetry=$(symmetry_mode)")
    println("faces=$(length(mesh.faces)) p1_dofs=$(p1.global_dof_count) dp0_dofs=$(dp0.global_dof_count) " *
            "drives=$(drive_count) steps=$(steps) threads=$(Threads.nthreads())")
    println("device=$(metal.device().name) regular_kernel=$(BeatEngineCore._normalized_metal_regular_kernel_mode()) " *
            "singular=$(singular_mode)")
    flush(stdout)

    Random.seed!(20260902)
    q_neumann = ComplexF32.(randn(Float32, dp0.global_dof_count, drive_count),
                            randn(Float32, dp0.global_dof_count, drive_count))

    device_cache = build_metal_regular_assembly_cache(
        mesh, p1, dp0, rule; singular_order=singular_order, symmetry_mode=symmetry_mode,
    )
    device_singular_cache = build_metal_singular_correction_cache(singular_cache)
    identity_cache = build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)

    # The two arms assemble the same physics through the two code paths the
    # solver itself can take, and each returns the host arrays to compare.
    fused_assemble = k -> begin
        system = assemble_burton_miller_neumann_system_metal(
            mesh, p1, dp0, q_neumann, k, rule;
            device_cache=device_cache,
            singular_cache=singular_cache,
            device_singular_cache=device_singular_cache,
            identity_cache=identity_cache,
            singular_order=singular_order,
            symmetry_mode=symmetry_mode,
        )
        try
            return Array(system.matrix)
        finally
            release_metal_burton_miller_system!(system)
        end
    end
    operator_assemble = k -> begin
        operators = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:metal,
            device_cache=device_cache,
            return_device=true,
            accelerator_quadrature=true,
            singular_cache=singular_cache,
            device_singular_cache=_normalized_metal_singular_mode_is_native() ? device_singular_cache : nothing,
            symmetry_mode=symmetry_mode,
        )
        try
            return Array(operators.hypersingular)
        finally
            release_operator_storage!(operators)
        end
    end

    failures = 0
    restore_singular_mode = get(ENV, "BLAB_METAL_SINGULAR_MODE", nothing)
    try
        # Arm 1, deterministic. Host singular corrections make the four-operator
        # assembly bit-reproducible, so this is the arm that can actually detect
        # a race: anything the producer shares with the consumer, or with its own
        # previous step, shows up as a changed bit.
        #
        # Arm 2, shipped. The fused path's native singular scatter accumulates
        # with atomics and is not bit-reproducible even sequentially, so the only
        # honest check is that pipelining does not widen the spread it already
        # has. The band is generous on purpose: floor and spread are two samples
        # of the same quantity, and a *new* divergence -- a raced scratch buffer
        # -- is orders of magnitude away, not a factor of two.
        arms = (
            (name="deterministic (four-operator, host singular)",
             singular="host", assemble=operator_assemble, require_bit_exact=true, band=1.0),
            (name="shipped (fused Burton-Miller, native singular)",
             singular="native", assemble=fused_assemble, require_bit_exact=false, band=4.0),
        )
        for arm in arms
            ENV["BLAB_METAL_SINGULAR_MODE"] = arm.singular
            println("\n--- arm: $(arm.name) ---")
            sequential_a = [arm.assemble(k) for k in wavenumbers]
            sequential_b = [arm.assemble(k) for k in wavenumbers]
            floor_spread = maximum(relative_difference(sequential_b[i], sequential_a[i]) for i in 1:steps)
            bit_exact = all(sequential_b[i] == sequential_a[i] for i in 1:steps)
            println("sequential run-to-run: bit-identical=$(bit_exact)  max relative spread=$(floor_spread)")
            flush(stdout)
            if arm.require_bit_exact && !bit_exact
                failures += 1
                println("FAILED: this arm is supposed to be bit-reproducible sequentially; " *
                        "the pipeline check below cannot mean anything without that.")
            end

            for depth in (1, 2, 3, 4)
                pipeline = start_sweep_assembly_pipeline(
                    index -> (index=index, k=wavenumbers[index], matrix=arm.assemble(wavenumbers[index])),
                    steps,
                    depth,
                    _ -> nothing,
                )
                mismatched = 0
                spread = 0.0
                identical = 0
                try
                    for index in 1:steps
                        produced = take_sweep_assembly!(pipeline, index)
                        produced.index == index && produced.k == wavenumbers[index] || (mismatched += 1)
                        spread = max(spread, relative_difference(produced.matrix, sequential_a[index]))
                        produced.matrix == sequential_a[index] && (identical += 1)
                    end
                finally
                    shutdown_sweep_assembly_pipeline!(pipeline, _ -> nothing)
                end
                ok = mismatched == 0 &&
                    (arm.require_bit_exact ? identical == steps : spread <= arm.band * floor_spread)
                ok || (failures += 1)
                println("depth=$(depth): order/pairing mismatches=$(mismatched)  " *
                        "bit-identical steps=$(identical)/$(steps)  " *
                        "max relative spread vs sequential=$(spread)  $(ok ? "OK" : "FAILED")")
                flush(stdout)
            end
        end
        ENV["BLAB_METAL_SINGULAR_MODE"] = "native"

        available = metal_sweep_memory_available()
        system_bytes = sizeof(ComplexF32) * (p1.global_dof_count^2 + p1.global_dof_count * drive_count)
        println("\nsystem=$(round(system_bytes / 1024^2; digits=2)) MiB  " *
                "device available=$(round(available / 1024^3; digits=2)) GiB  " *
                "derived depth=$(sweep_pipeline_depth(system_bytes, available, steps))")

        # Cancellation: stop part way, and account for every step that was built.
        produced_count = Threads.Atomic{Int}(0)
        released_count = Threads.Atomic{Int}(0)
        note_release = _ -> (Threads.atomic_add!(released_count, 1); nothing)
        pipeline = start_sweep_assembly_pipeline(
            index -> begin
                Threads.atomic_add!(produced_count, 1)
                (index=index, matrix=fused_assemble(wavenumbers[index]))
            end,
            steps,
            4,
            note_release,
        )
        consumed = 0
        for index in 1:2
            take_sweep_assembly!(pipeline, index)
            consumed += 1
        end
        drained = shutdown_sweep_assembly_pipeline!(pipeline, note_release)
        accounted = produced_count[] == consumed + released_count[]
        stopped_early = produced_count[] < steps
        accounted && stopped_early || (failures += 1)
        println("cancel: produced=$(produced_count[]) consumed=$(consumed) released=$(released_count[]) " *
                "drained=$(drained)  every step accounted for=$(accounted)  stopped early=$(stopped_early)")
    finally
        release_metal_fused_identity_cache!(identity_cache)
        release_metal_singular_correction_cache!(device_singular_cache)
        if restore_singular_mode === nothing
            delete!(ENV, "BLAB_METAL_SINGULAR_MODE")
        else
            ENV["BLAB_METAL_SINGULAR_MODE"] = restore_singular_mode
        end
        release_metal_regular_assembly_cache!(device_cache)
    end

    failures > 0 && error("Metal sweep pipeline validation failed in $(failures) check(s).")
    println("\nPASS: pipelined assembly is in order, correctly paired, and no less reproducible " *
            "than the sequential path.")
    return nothing
end

validate_metal_sweep_pipeline()
