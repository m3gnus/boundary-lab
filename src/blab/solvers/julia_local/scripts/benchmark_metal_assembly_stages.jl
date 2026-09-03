# Where does a fused Burton-Miller assembly spend its time at small N?
using Printf
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
regular_order = parse(Int, get(ENV, "ASM_BENCH_ORDER", "4"))
rule = triangle_rule(T, regular_order)
singular_order = 4
@printf("%s: %d faces, %d p1 dofs, symmetry %s\n", basename(mesh_path), length(mesh.faces), p1.global_dof_count, sym)

sing = build_singular_correction_cache(mesh, singular_order)
dev = build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=singular_order, symmetry_mode=sym)
dsing = build_metal_singular_correction_cache(sing)
id_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=sym)
id_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=sym)
idc = build_metal_fused_identity_cache(id_p1_p1, id_p1_dp0, T)
qcols = ComplexF32.(randn(T, dp0.global_dof_count, 1), randn(T, dp0.global_dof_count, 1))

k = T(2pi * 2000.0 / 343.0)
timing = Dict{String,Float64}()
# warm
sysm = assemble_burton_miller_neumann_system_metal(mesh, p1, dp0, qcols, k, rule;
    device_cache=dev, singular_cache=sing, device_singular_cache=dsing,
    identity_cache=idc, singular_order=singular_order, symmetry_mode=sym, timing=timing)
release_metal_burton_miller_system!(sysm)

best = Dict{String,Float64}(); walls = Float64[]
for _ in 1:6
    empty!(timing)
    t = time()
    s = assemble_burton_miller_neumann_system_metal(mesh, p1, dp0, qcols, k, rule;
        device_cache=dev, singular_cache=sing, device_singular_cache=dsing,
        identity_cache=idc, singular_order=singular_order, symmetry_mode=sym, timing=timing)
    metal.synchronize()
    push!(walls, time() - t)
    release_metal_burton_miller_system!(s)
    for (kk, v) in timing
        best[kk] = min(get(best, kk, Inf), v)
    end
end
@printf("\n  assembly wall: min %.2f ms  (median %.2f)\n", 1e3*minimum(walls), 1e3*sort(walls)[3])
for (kk, v) in sort(collect(best), by = x -> -x[2])
    @printf("    %-40s %7.2f ms  %5.1f%%\n", kk, 1e3*v, 100*v/minimum(walls))
end
