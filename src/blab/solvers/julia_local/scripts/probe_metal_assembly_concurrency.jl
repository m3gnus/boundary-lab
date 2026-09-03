# Ceiling probe: do N concurrent fused assemblies cost less than N sequential ones?
#
# Metal.jl command queues are task-local, so each worker task is spawned ONCE and
# reused for every frequency it handles -- the 2026-09-03 finding was that a task
# per frequency rebuilds the queue and loses more than the overlap wins.
# If concurrency is worth building, wall(N concurrent) / wall(1) must be well
# under N. If it is ~N the GPU is already saturated and no scheduler can help.
using Printf, Base.Threads
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
metal = BeatEngineCore.METAL_MODULE
metal === nothing && error("Run with the julia_metal project.")

mesh_path = ENV["ASM_BENCH_MESH"]
T = Float32
scale = T(parse(Float64, get(ENV, "ASM_BENCH_SCALE", "0.001")))
sym = Symbol(get(ENV, "ASM_BENCH_SYM", "xy"))
mesh = snap_symmetry_planes(load_gmsh22_with_tags(mesh_path, scale), sym)
p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh)
order = parse(Int, get(ENV, "ASM_BENCH_ORDER", "4"))
rule = triangle_rule(T, order); singular_order = 4
sing = build_singular_correction_cache(mesh, singular_order)
dev = build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=singular_order, symmetry_mode=sym)
dsing = build_metal_singular_correction_cache(sing)
idc = build_metal_fused_identity_cache(
    assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=sym),
    assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=sym), T)
q = ComplexF32.(randn(T, dp0.global_dof_count, 1), randn(T, dp0.global_dof_count, 1))
@printf("%s: %d faces, %d p1 dofs, sym %s, %d julia threads\n",
        basename(mesh_path), length(mesh.faces), p1.global_dof_count, sym, nthreads())

one_assembly(k) = begin
    s = assemble_burton_miller_neumann_system_metal(mesh, p1, dp0, q, k, rule;
        device_cache=dev, singular_cache=sing, device_singular_cache=dsing,
        identity_cache=idc, singular_order=singular_order, symmetry_mode=sym)
    release_metal_burton_miller_system!(s)
end
ks = [T(2pi * f / 343.0) for f in (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0)]
one_assembly(ks[1]); metal.synchronize()

base = minimum(begin; t=time(); one_assembly(ks[1]); metal.synchronize(); time()-t; end for _ in 1:5)
@printf("\n  single assembly: %6.1f ms\n\n", 1e3*base)
@printf("  %5s | %9s %9s %9s %7s\n", "N", "seq (ms)", "conc (ms)", "ideal", "speedup")
for n in (2, 3, 4)
    n > length(ks) && continue
    seq = minimum(begin
        t=time(); for i in 1:n; one_assembly(ks[i]); end; metal.synchronize(); time()-t
    end for _ in 1:3)
    conc = minimum(begin
        t=time()
        tasks = [Threads.@spawn begin one_assembly(ks[i]); metal.synchronize(); end for i in 1:n]
        foreach(wait, tasks)
        time()-t
    end for _ in 1:3)
    @printf("  %5d | %9.1f %9.1f %9.1f %6.2fx\n", n, 1e3*seq, 1e3*conc, 1e3*base, seq/conc)
end
