# Field-evaluation occupancy: the chunked kernel against the one-thread-per-point
# original, and both against the CPU path. Run with the julia_metal project.
using Printf
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

metal = BeatEngineCore.METAL_MODULE
metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
const Metal = metal

mesh_path = get(ENV, "FIELD_BENCH_MESH", joinpath(@__DIR__, "..", "test_meshes", "sample.msh"))
T = Float32
scale = T(parse(Float64, get(ENV, "FIELD_BENCH_SCALE", "0.001")))
mesh = load_gmsh22_with_tags(mesh_path, scale)
rule = triangle_rule(T, parse(Int, get(ENV, "FIELD_BENCH_ORDER", "4")))
cache_cpu = build_field_evaluation_cache(mesh, rule)
cache_gpu = build_metal_field_evaluation_cache(cache_cpu)
n_src = cache_gpu.source_count
@printf("mesh %s: %d vertices, %d triangles, %d field sources\n",
        basename(mesh_path), length(mesh.vertices), length(mesh.faces), n_src)

pressure = ComplexF32.(randn(T, length(mesh.vertices)) .+ im .* randn(T, length(mesh.vertices)))
q_neumann = ComplexF32.(randn(T, length(mesh.faces)) .+ im .* randn(T, length(mesh.faces)))
k = T(2pi * 2000.0 / 343.0)

function polar_points(n)
    pts = Vector{NTuple{3,T}}()
    for plane in (0, 1), i in 0:(n - 1)
        a = T(pi) * T(i) / T(max(1, n - 1))
        s, c = sin(a), cos(a)
        push!(pts, plane == 0 ? (T(2) * s, T(0), T(2) * c) : (T(0), T(2) * s, T(2) * c))
    end
    return pts
end

best(f, r) = minimum(begin; t = time(); f(); Metal.synchronize(); time() - t; end for _ in 1:r)

for npts in (74, 296, 2664)
    pts = polar_points(npts ÷ 2)
    ENV["BLAB_METAL_FIELD_CHUNKS"] = "1"
    ref = evaluate_galerkin_field_metal(pts, mesh, pressure, q_neumann, k, cache_gpu)
    t_old = best(() -> evaluate_galerkin_field_metal(pts, mesh, pressure, q_neumann, k, cache_gpu), 5)
    delete!(ENV, "BLAB_METAL_FIELD_CHUNKS")
    got = evaluate_galerkin_field_metal(pts, mesh, pressure, q_neumann, k, cache_gpu)
    t_new = best(() -> evaluate_galerkin_field_metal(pts, mesh, pressure, q_neumann, k, cache_gpu), 5)
    cpu = evaluate_galerkin_field_cpu(pts, mesh, pressure, q_neumann, k, cache_cpu)
    chunks = BeatEngineCore._metal_field_chunk_count(length(pts), n_src)
    rel(a, b) = maximum(abs.(a .- b)) / max(eps(T), maximum(abs.(b)))
    @printf("  %5d points  chunks %4d   old %7.2f ms   new %7.2f ms   %5.2fx   |new-old| %.3e  |new-cpu| %.3e  |old-cpu| %.3e\n",
            length(pts), chunks, 1e3 * t_old, 1e3 * t_new, t_old / t_new,
            rel(got, ref), rel(got, cpu), rel(ref, cpu))
end
