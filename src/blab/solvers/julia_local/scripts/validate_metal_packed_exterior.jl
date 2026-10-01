# Ownership and multi-drive gate for the packed exterior kernels.
using Test, LinearAlgebra, Random, StaticArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
const Engine = BeatEngineCore
const metal = Engine.METAL_MODULE
metal !== nothing && metal.functional() || error("A functional Metal device is required.")
Threads.nthreads() > 1 || error("Run with at least two Julia threads.")

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "two_tetrahedra.msh"), 1.0f0)
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float32, 4)
singular = build_singular_correction_cache(mesh, 4)
cache = build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=4)
dsingular = build_metal_singular_correction_cache(singular)
identity = Engine.build_metal_fused_identity_cache(
    assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1),
    assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0), Float32,
)
field = build_metal_field_evaluation_cache(mesh, rule)
Random.seed!(20260930)
qs = [complex.(randn(Float32, dp0.global_dof_count, 2), randn(Float32, dp0.global_dof_count, 2)) for _ in 1:2]
ks = Float32[2pi * 200 / 343, 2pi * 2106 / 343]
function assemble(index)
    # Race locked publication of packed singular geometry without atomic scatter.
    Engine._metal_fused_singular_tables_for(cache, dsingular)
    system = assemble_burton_miller_neumann_system_metal(mesh, p1, dp0, qs[index], ks[index], rule;
        skip_singular=true, device_cache=cache, singular_cache=singular, device_singular_cache=dsingular, identity_cache=identity)
    try
        return (Array(system.matrix), Array(system.rhs))
    finally
        release_metal_burton_miller_system!(system)
    end
end
try
    @testset "shared geometry, independent assembly scratch" begin
        # Race the first lazy table creation as well as warm table reuse.
        concurrent = fetch.([Threads.@spawn assemble(i) for i in 1:2])
        sequential = [assemble(i) for i in 1:2]
        @test concurrent == sequential
        @test !hasproperty(cache.fused_gather_tables[], :blocks)
        for _ in 1:3
            @test fetch.([Threads.@spawn assemble(i) for i in 1:2]) == sequential
        end
    end
    @testset "packed multi-drive field" begin
        pressures = [complex.(randn(Float32, length(mesh.vertices)), randn(Float32, length(mesh.vertices))) for _ in 1:9]
        neumanns = [complex.(randn(Float32, length(mesh.faces)), randn(Float32, length(mesh.faces))) for _ in 1:9]
        points = [SVector{3,Float32}(3, 0, 2), SVector{3,Float32}(0, 3, 2)]
        for nd in (0, 1, 2, 8, 9)
            expected = [Engine.evaluate_galerkin_field_metal(points, mesh, pressures[i], neumanns[i], ks[1], field) for i in 1:nd]
            actual = Engine.evaluate_galerkin_field_metal_multi(points, mesh, pressures[1:nd], neumanns[1:nd], ks[1], field)
            @test length(actual) == nd
            # ND specializations can round differently under GPU fast math. Bound
            # that difference, while requiring each specialization to repeat exactly.
            relative_l2 = nd == 0 ? 0.0 : maximum(
                norm(ComplexF64.(a) - ComplexF64.(b)) / max(norm(ComplexF64.(b)), eps(Float64))
                for (a, b) in zip(actual, expected)
            )
            max_db = nd == 0 ? 0.0 : maximum(
                abs(20 * log10(abs(ComplexF64(x)) / abs(ComplexF64(y))))
                for (a, b) in zip(actual, expected) for (x, y) in zip(a, b)
            )
            println("multi_drive nd=$nd relative_l2=$relative_l2 max_abs_db=$max_db")
            @test relative_l2 <= 8eps(Float32)
            @test max_db <= 1e-5
            @test actual == Engine.evaluate_galerkin_field_metal_multi(points, mesh, pressures[1:nd], neumanns[1:nd], ks[1], field)
        end
        for eval_points in (points, SVector{3,Float32}[]), nd in (1, 2)
            @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi(eval_points, mesh, pressures[1:nd], neumanns[1:nd-1], ks[1], field)
            @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi(eval_points, mesh, [pressures[1][1:end-1]], [neumanns[1]], ks[1], field)
            @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi(eval_points, mesh, [pressures[1]], [neumanns[1][1:end-1]], ks[1], field)
        end
        # The Float64 and empty-point fallback must validate before zip can truncate.
        mesh64 = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "two_tetrahedra.msh"), 1.0)
        empty_cache = Engine.MetalFieldEvaluationCache{Float64}(nothing, nothing, nothing, nothing, nothing, nothing, 0)
        @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi([], mesh64, pressures[1:2], neumanns[1:1], 1.0, empty_cache)
        @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi([], mesh64, [pressures[1][1:end-1]], [neumanns[1]], 1.0, empty_cache)
    end
finally
    release_metal_field_evaluation_cache!(field)
    Engine.release_metal_fused_identity_cache!(identity)
    release_metal_singular_correction_cache!(dsingular)
    release_metal_regular_assembly_cache!(cache)
end
