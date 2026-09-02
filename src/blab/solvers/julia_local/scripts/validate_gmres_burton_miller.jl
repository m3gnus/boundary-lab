# Equivalence and iteration-count gate for the adaptive dense solve.
#
# The unit tests in `tests/runtests.jl` check the Krylov invariants on
# synthetic systems that converge in about 25 iterations. This script runs the
# same checks against a real assembled Burton-Miller operator, because the
# defects this path has actually had — a Float32 Arnoldi recurrence losing
# orthogonality, and a breakdown test that read the Hessenberg subdiagonal
# after the Givens rotation had zeroed it — both produce a plausible-looking
# iteration count several times too large, and neither is visible on a problem
# that finishes in ten steps.
#
# That is the general form of the lesson: a numerical routine validated only
# on cases that converge quickly is not validated for the cases it exists for.
#
# What is checked:
#
#  1. GMRES agrees with the dense LU to Float32 noise, per drive.
#  2. The true relative residual meets the tolerance — the tolerance is on
#     ||b - Ax||/||b|| recomputed against the operator, not on the
#     preconditioned residual the Givens recursion tracks.
#  3. Three independent Krylov variants agree on the iteration count:
#     Float64 space, Float32 with unconditional reorthogonalization, and
#     Float32 with DGKS. Agreement between independent remedies is the
#     evidence; no single count is.
#  4. The count does not depend on the restart length, for restarts above the
#     count. A converging solver does not care what the restart is; an
#     iteration count that tracks it is not an iteration count.
#  5. Plain Float32 single Gram-Schmidt is materially worse, which is what
#     makes check 3 meaningful rather than vacuous. The failing row is in the
#     output on purpose: three variants agreeing means something only because
#     a fourth fails on the same operator.
#
# Every check runs across the band rather than at one frequency, and which end
# is hard depends on the mesh, so do not "simplify" this to a single point.
# Two mechanisms pull in opposite directions:
#
#  * Low-k conditioning. The uncapped Burton-Miller coupling `eta = i/k` grows
#    without bound as k goes to zero, so the bottom of the band is hard. This
#    dominates on well-resolved meshes: on the ATH ladder's A5 (5,107 dofs) the
#    counts fall monotonically across the band.
#  * High kh. A coarse mesh runs out of elements per wavelength, so the top of
#    the band is hard, which pulls the other way.
#
# Which end is hard is therefore a property of the mesh, and a gate at mid-band
# alone would pass straight through either mechanism. Guarding the low-k
# coupling path specifically needs a ladder-sized mesh via
# BLAB_VALIDATE_MESH_PATH; the bundled fixture cannot exhibit that failure at
# all. Same principle as the iteration floor below: put the gate against the
# hard corner of the configuration it actually runs in.
#
# The counts this paragraph used to quote -- 44 / 51 / 63 on the bundled sample
# and 70 / 51 / 51 on A5 -- were measured before `8aa2539`, with the random
# right-hand side this gate no longer uses. On the physical drive the bundled
# sample reads 53 / 58 / 55 at 500 / 2000 / 6000 Hz, which is not the same
# ordering, so the old numbers did not merely shift. **Never quote an iteration
# count without the drive that produced it.** The random-versus-physical gap is
# about 1.6x on these meshes and it misled three sessions in one day, including
# by surviving a rebase inside a single branch: the number was written down
# under one drive and read back under another with nothing in its appearance
# changed.
#
# The capped coupling (`eta = i|k|/max(k^2, c)`, c = 1/R^2 from the body's own
# bounding box) addresses the first mechanism and must not touch the second.
# The final section gates exactly that, and it picks its own low frequency from
# the mesh -- a quarter of the cap's engagement frequency -- because the
# engagement frequency is geometry-dependent and a hard-coded low frequency
# would be inside the capped regime on one fixture and outside it on the next.
# Two properties are checked, and the first matters more than the second:
#
#  1. Inertness. Above the engagement wavenumber the capped operator is
#     *identical* to the uncapped one, entry for entry, not merely close. That
#     is what makes "the mid and high band are unchanged" a proof rather than a
#     tolerance, and it is why the cap is written to evaluate `inv(|k|)` on that
#     branch instead of the algebraically equal `|k| / k^2`.
#  2. The iteration count below the engagement frequency does not get worse.
#     Measured it improves -- 78 to 56 on A5 at 100 Hz, 80 to 44 on A1 -- but
#     the gate is one-sided on purpose: the cap exists to remove a conditioning
#     hazard, and a cap that merely fails to help is not a regression, while one
#     that hurts is.
#
#   BLAB_VALIDATE_MESH_PATH   absolute mesh path (default: the bundled sample)
#   BLAB_VALIDATE_SCALE       mesh scale (default 0.001 for the sample)
#   BLAB_VALIDATE_SYMMETRY    off | x | xy
#   BLAB_VALIDATE_DRIVES      independent drive columns (default 2)
#   BLAB_VALIDATE_FREQUENCY_HZ  default 2000
#   BLAB_BEAT_BM_COUPLING_CAP   auto (default) | off | c in 1/m^2
using LinearAlgebra, Printf, Random

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const KRYLOV_VARIANTS = (
    ("float64 dgks", ComplexF64, :dgks),
    ("float32 always", ComplexF32, :always),
    ("float32 dgks", ComplexF32, :dgks),
)

function validate_gmres_burton_miller()
    mesh_path = get(ENV, "BLAB_VALIDATE_MESH_PATH", "")
    scale = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_SCALE", isempty(mesh_path) ? "0.001" : "1.0")))
    if isempty(mesh_path)
        mesh_path = joinpath(@__DIR__, "..", "test_meshes", get(ENV, "BLAB_VALIDATE_MESH", "sample.msh"))
    end
    symmetry_mode = Symbol(get(ENV, "BLAB_VALIDATE_SYMMETRY", "off"))
    drive_count = parse(Int, get(ENV, "BLAB_VALIDATE_DRIVES", "2"))
    regular_order = parse(Int, get(ENV, "BLAB_VALIDATE_REGULAR_ORDER", "4"))
    singular_order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "4"))
    frequencies = Float32.(parse.(Float64, split(get(ENV, "BLAB_VALIDATE_FREQUENCY_HZ", "500,2000,6000"), ",")))
    tolerance = parse(Float64, get(ENV, "BLAB_VALIDATE_GMRES_TOLERANCE", "1e-6"))
    agreement_tolerance = parse(Float64, get(ENV, "BLAB_VALIDATE_GMRES_AGREEMENT", "1e-4"))

    mesh = load_gmsh22_with_tags(mesh_path, scale)
    mesh = snap_symmetry_planes(mesh, symmetry_mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, regular_order)
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)

    n = p1.global_dof_count
    sound_speed = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_SOUND_SPEED", "343.0")))
    body_radius = burton_miller_body_radius(mesh; symmetry_mode=symmetry_mode)
    coupling_cap = burton_miller_coupling_cap(mesh; symmetry_mode=symmetry_mode)
    cap_hz = coupling_cap > 0 ? sound_speed * sqrt(coupling_cap) / Float32(2pi) : 0.0f0
    println("fixture=$(mesh_path) scale=$(scale) symmetry=$(symmetry_mode)")
    println("faces=$(length(mesh.faces)) p1_dofs=$n frequencies=$(frequencies) drives=$(drive_count)")
    @printf("body_radius=%.4f m coupling_cap=%.4f 1/m^2 (engages below %.1f Hz)\n",
            body_radius, coupling_cap, cap_hz)
    flush(stdout)

    failures = String[]
    check(condition, message) = condition || push!(failures, message)

    # The physical excitation the solver actually applies: unit normal velocity
    # on the driver tag, q = i rho omega, with a per-drive phase so the columns
    # are independent.
    #
    # This gate used a random right-hand side and was materially easier than
    # the path it guards. On the sliver-rim ATH meshes a random drive converges
    # in 38-113 iterations while the localised tag-2 drive needs 141-212 and
    # falls back to the LU at the bottom of the band -- so the gate reported
    # health on exactly the configuration that fails in production. A gate
    # easier than production passes changes that break production.
    function physical_drive(k::Float32)
        q = zeros(ComplexF32, dp0.global_dof_count, drive_count)
        coefficient = ComplexF32(0, 1) * 1.2041f0 * k * 343.0f0
        @inbounds for index in eachindex(mesh.faces)
            mesh.physical_tags[index] == 2 || continue
            for drive in 1:drive_count
                q[dp0.local_to_global[index], drive] = coefficient * cis(Float32(0.7 * (drive - 1)))
            end
        end
        # A fixture without a tag-2 driver is driven uniformly rather than
        # silently solved with a zero right-hand side.
        all(iszero, q) && (q .= coefficient)
        return q
    end
    for frequency_hz in frequencies
    k = Float32(2pi) * frequency_hz / 343.0f0
    println()
    println("--- $(frequency_hz) Hz ---")
    q_neumann = physical_drive(k)
    # The shipped configuration, cap included: what is gated is what runs.
    system = assemble_burton_miller_neumann_system_cpu(
        mesh, p1, dp0, q_neumann, k, rule;
        identity_p1_p1=identity_p1_p1, identity_p1_dp0=identity_p1_dp0,
        skip_singular=false, singular_order=singular_order,
        singular_cache=singular_cache, symmetry_mode=symmetry_mode,
        coupling_cap=coupling_cap,
    )
    matrix = system.matrix
    rhs = system.rhs

    reference = lu(copy(matrix)) \ rhs
    reference_scale = max(norm(reference), eps(Float32))

    gmres_solution, report = beat_solve_dense_system(matrix, rhs; method=:gmres)
    agreement = norm(gmres_solution - reference) / reference_scale
    @printf("gmres_vs_lu_relative=%.3e  iterations=%s  residuals=%s\n",
            agreement, report.iterations, report.relative_residuals)
    # A fallback is not a failure. On the sliver-rim meshes GMRES genuinely
    # cannot reach the tolerance at the bottom of the band, and degrading to
    # the LU is the designed and correct response -- it is what
    # hornlab-metal-bem lacks, where the same geometry returns info=-999 and
    # kills the sweep. What the gate must check is that the answer is right
    # either way, so the fallback is reported and the *solution* is asserted.
    report.fell_back &&
        println("note: GMRES did not converge and fell back to the dense LU (expected on " *
                "sliver-rim meshes at low frequency); the solution is checked below either way")
    check(agreement <= agreement_tolerance,
          "$(frequency_hz) Hz: solution disagrees with the LU by $(agreement), above $(agreement_tolerance)")
    # Recomputed independently of the solver, which evaluates `b - Ax` as one
    # fused gemv accumulating into b. Forming `Ax` in full and subtracting
    # afterwards cancels, and in Float32 that costs about sqrt(N) * eps of the
    # result -- 1.1e-5 at 7,890 dofs, above the 1e-6 the solver is asked for.
    # So the bound is the larger of the tolerance and that evaluation floor:
    # below the floor this quantity is measuring Float32 subtraction, not the
    # solve. The solver's own fused evaluation is the more accurate of the two,
    # and the check that the answer is actually right is the agreement with the
    # LU above, not this one.
    evaluation_floor = sqrt(Float64(n)) * eps(Float32)
    residual_bound = max(4 * tolerance, evaluation_floor)
    for drive in 1:drive_count
        report.fell_back && break  # the LU's residual, not the Krylov path's
        residual = norm(matrix * view(gmres_solution, :, drive) - view(rhs, :, drive)) /
            max(norm(view(rhs, :, drive)), eps(Float32))
        check(residual <= residual_bound,
              "$(frequency_hz) Hz drive $drive: recomputed relative residual $(residual) " *
              "exceeds $(residual_bound) (4x tolerance, or the Float32 evaluation floor)")
    end

    # The routing decision itself must be reported, not silently taken.
    plan = report.plan
    @printf("plan: method=%s reason=%s lu_model=%.4f s gmres_model=%.4f s\n",
            plan.method, plan.reason, plan.lu_model_seconds, plan.gmres_model_seconds)

    preconditioner = beat_diagonal_preconditioner(matrix)
    single_drive_rhs = Vector{ComplexF32}(view(rhs, :, 1))
    counts = Dict{String,Int}()
    println()
    @printf("%-16s %10s %12s %12s\n", "variant", "iterations", "residual", "vs LU")
    for (label, krylov_type, reorthogonalize) in KRYLOV_VARIANTS
        x = zeros(ComplexF32, n)
        result = beat_gmres!(x, matrix, copy(single_drive_rhs);
                             krylov_type=krylov_type, reorthogonalize=reorthogonalize,
                             preconditioner=preconditioner, tolerance=tolerance)
        difference = norm(x - view(reference, :, 1)) / max(norm(view(reference, :, 1)), eps(Float32))
        @printf("%-16s %10d %12.3e %12.3e%s\n", label, result.iterations,
                result.relative_residual, difference, result.converged ? "" : "  NOT CONVERGED")
        # Not asserted: on these meshes non-convergence is a property of the
        # operator, and all three variants agreeing on *where* they stop is the
        # evidence this gate exists for.
        counts[label] = result.iterations
    end

    baseline = counts["float64 dgks"]
    check(baseline >= 10,
          "$(frequency_hz) Hz: the reference variant converged in $baseline iterations, too few to exercise " *
          "orthogonality loss; use a harder mesh or frequency for this gate to mean anything")
    for (label, _, _) in KRYLOV_VARIANTS
        spread = abs(counts[label] - baseline)
        check(spread <= max(2, baseline ÷ 10),
              "$(frequency_hz) Hz: $label needed $(counts[label]) iterations against $baseline for the Float64 space")
    end

    # Restart independence, above the converged count.
    println()
    for restart in (baseline + 10, 2 * baseline + 50, 0)
        x = zeros(ComplexF32, n)
        result = beat_gmres!(x, matrix, copy(single_drive_rhs);
                             preconditioner=preconditioner, restart=restart, tolerance=tolerance)
        @printf("restart=%-6d iterations=%d\n", restart, result.iterations)
        check(result.iterations == baseline,
              "$(frequency_hz) Hz: restart $restart changed the count to $(result.iterations) from $baseline; " *
              "a converging solver does not depend on the restart above its own count")
    end

    # And the failure mode the remedies exist for must be reachable, or their
    # agreement above proves nothing.
    # Capped well above the healthy count: the question is only whether the
    # unreorthogonalised variant is much worse, not by exactly how much, and
    # letting it run to a 1,000-iteration cap costs more than the rest of the
    # gate put together on a 7,890-dof mesh.
    stalled_cap = 3 * baseline + 50
    x = zeros(ComplexF32, n)
    stalled = beat_gmres!(x, matrix, copy(single_drive_rhs);
                          krylov_type=ComplexF32, reorthogonalize=:never,
                          preconditioner=preconditioner, tolerance=tolerance,
                          max_iterations=stalled_cap)
    println()
    @printf("float32 single MGS: %d iterations against %d%s\n",
            stalled.iterations, baseline,
            stalled.converged ? "" : " (did not converge -- the failure the remedies cover)")
    # Warned, not asserted, for the same reason as the unit-test counterpart:
    # whether a Float32 recurrence loses orthogonality is a property of the
    # host's arithmetic rather than of this code, and a suite that goes red on
    # a machine where the solver is correct is reporting the wrong thing. The
    # margin here is normally enormous -- 1000 against 51 on an M1 Max -- so a
    # small one is worth surfacing loudly.
    if stalled.iterations <= 2 * baseline
        println("WARNING: $(frequency_hz) Hz: single Gram-Schmidt took $(stalled.iterations) " *
                "against $baseline for the Float64 space. The failure the remedies exist for " *
                "is barely reachable on this host, so their agreement proves less here.")
    end
    end

    # ---------------------------------------------------------------- cap ---
    #
    # Everything above ran at the shipped cap. This section is about what the
    # cap does and does not touch, and it needs the uncapped operator to say
    # so, which is why it assembles a second time rather than reusing the loop.
    println()
    println("--- coupling cap ---")

    # The sign is the one thing here that a solve cannot catch: both signs give
    # a consistent, solvable system, and the wrong one costs ~900 GMRES
    # iterations where the right one costs under 50 (Marburg). Gate it directly.
    for probe in (0.5f0, 9.16f0, 109.9f0)
        check(real(burton_miller_coupling(probe, coupling_cap)) == 0,
              "coupling at k=$probe is not purely imaginary")
        check(imag(burton_miller_coupling(probe, coupling_cap)) > 0,
              "coupling at k=$probe has the wrong sign for e^{-i omega t}; " *
              "eta must be +i * (positive real)")
    end
    check(burton_miller_coupling(9.16f0, 0) === ComplexF32(0, 1) / 9.16f0,
          "the uncapped coupling is no longer the bare i/k, bit for bit")

    if coupling_cap <= 0
        println("cap disabled by $(BeatEngineCore.BEAT_BM_COUPLING_CAP_ENV); inertness and " *
                "low-frequency checks skipped")
    else
        k_engage = sqrt(coupling_cap)

        # 1. Inertness, as an identity rather than a tolerance. Checked at the
        #    lowest gated frequency above the engagement wavenumber -- the one
        #    closest to the transition, and so the only one where a mistake in
        #    the branch could hide. Every frequency would cost two more full
        #    assemblies each on a ladder-sized mesh for no more evidence.
        inert_probes = filter(f -> Float32(2pi) * f / sound_speed >= k_engage, frequencies)
        for frequency_hz in (isempty(inert_probes) ? Float32[] : [minimum(inert_probes)])
            k_probe = Float32(2pi) * frequency_hz / sound_speed
            probe_drive = physical_drive(k_probe)
            capped = assemble_burton_miller_neumann_system_cpu(
                mesh, p1, dp0, probe_drive, k_probe, rule;
                identity_p1_p1=identity_p1_p1, identity_p1_dp0=identity_p1_dp0,
                skip_singular=false, singular_order=singular_order,
                singular_cache=singular_cache, symmetry_mode=symmetry_mode,
                coupling_cap=coupling_cap,
            )
            bare = assemble_burton_miller_neumann_system_cpu(
                mesh, p1, dp0, probe_drive, k_probe, rule;
                identity_p1_p1=identity_p1_p1, identity_p1_dp0=identity_p1_dp0,
                skip_singular=false, singular_order=singular_order,
                singular_cache=singular_cache, symmetry_mode=symmetry_mode,
                coupling_cap=0,
            )
            identical = capped.matrix == bare.matrix && capped.rhs == bare.rhs
            @printf("%.0f Hz (kR=%.2f): capped operator identical to uncapped: %s\n",
                    frequency_hz, k_probe / k_engage, identical)
            check(identical,
                  "$(frequency_hz) Hz is above the cap's engagement wavenumber but the capped " *
                  "operator differs from the uncapped one; the cap is not inert in the mid band")
        end

        # 2. Below the engagement frequency the cap must not cost iterations.
        #    The frequency is a quarter of the engagement point so that this
        #    holds on any fixture, not only the one it was written against.
        low_hz = 0.25f0 * cap_hz
        k_low = Float32(2pi) * low_hz / sound_speed
        counts = Int[]
        for cap in (coupling_cap, 0)
            low = assemble_burton_miller_neumann_system_cpu(
                mesh, p1, dp0, physical_drive(k_low), k_low, rule;
                identity_p1_p1=identity_p1_p1, identity_p1_dp0=identity_p1_dp0,
                skip_singular=false, singular_order=singular_order,
                singular_cache=singular_cache, symmetry_mode=symmetry_mode,
                coupling_cap=cap,
            )
            x = zeros(ComplexF32, n)
            result = beat_gmres!(x, low.matrix, Vector{ComplexF32}(view(low.rhs, :, 1));
                                 preconditioner=beat_diagonal_preconditioner(low.matrix),
                                 tolerance=tolerance)
            push!(counts, result.iterations)
            @printf("%.0f Hz cap=%-8s iterations=%d converged=%s\n",
                    low_hz, cap == 0 ? "off" : "on", result.iterations, result.converged)
        end
        check(counts[1] <= counts[2],
              "at $(low_hz) Hz the capped coupling needed $(counts[1]) iterations against " *
              "$(counts[2]) uncapped; the cap exists to improve low-frequency conditioning, " *
              "not to cost iterations")
    end

    println()
    if isempty(failures)
        println("PASS")
        return true
    end
    for failure in failures
        println("FAIL: $failure")
    end
    return false
end

validate_gmres_burton_miller() || exit(1)
