"""
    BeatEngineRocmBundle

A precompilable home for the BEAT engine's ROCm build.

The engine (`julia_local/src`) and the worker driver
(`julia_local/BeatEngineDriver.jl`) were, until this package existed, pulled
into `Main` with `include` on every worker start. `include` compiles from
source every time: neither Julia's pkgimage cache nor a PackageCompiler
sysimage can see code that is not in a package, which is why the sysimage in
`docker/` never captured the engine, and why a cold worker spent most of its
start-up compiling before it could accept a job.

Loading the same sources from a package fixes that, and fixes it without
anything that can go stale. Julia records every `include`d file in the cache
header and rechecks them on load, so editing an engine source invalidates the
pkgimage automatically and the next process rebuilds it. A hand-built sysimage
has no such property, which is why this is the primary mechanism and a
sysimage is an optional extra layered on top of it.

There is one package per accelerator because a package gets one precompile
cache and the engine chooses its backend while it loads. `BEAT_ENGINE_BACKEND`
is what `BeatEngineCore` consults in place of the process environment: the
environment is read when the cache is *built*, so a bundle has to state its
backend rather than inherit whatever the building process happened to have set.
"""
module BeatEngineRocmBundle

using PrecompileTools: @compile_workload

const BEAT_ENGINE_BACKEND = "rocm"

#: Where the engine and driver sources are, found rather than assumed.
#: Boundary Lab calls that directory `julia_local`; the Python wrapper that
#: vendors these sources flattens it to `julia`. Looking for the directory
#: that actually holds `BeatEngineCore.jl` means re-vendoring is a file copy
#: and not a path rewrite -- and a path rewrite that was missed would fail at
#: precompile time in the shipped package, which is the worst place to find it.
const ENGINE_DIR = let root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    found = nothing
    for name in ("julia_local", "julia")
        candidate = joinpath(root, name)
        if isfile(joinpath(candidate, "src", "BeatEngineCore.jl"))
            found = candidate
            break
        end
    end
    found === nothing && error("No BEAT engine sources found under $(root).")
    found
end

include(joinpath(ENGINE_DIR, "src", "BeatEngineCore.jl"))

using .BeatEngineCore

include(joinpath(ENGINE_DIR, "BeatEngineDriver.jl"))

#: Four triangles, one closed tetrahedron, every face tagged as the source.
#: The workload below only needs a mesh that solves; nothing it compiles
#: depends on the geometry.
const WORKLOAD_MESH = """
\$MeshFormat
2.2 0 8
\$EndMeshFormat
\$PhysicalNames
1
2 2 "warmup"
\$EndPhysicalNames
\$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
\$EndNodes
\$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
\$EndElements
"""

# Shared by the compiled CPU and Metal bundles. The plate touches both symmetry
# planes, has non-adjacent triangles, and has singular pairs with its images.
function workload_plate_mesh()
    io = IOBuffer()
    print(io, first(split(WORKLOAD_MESH, "\$Nodes")), "\$Nodes\n9\n")
    for y in 0:2, x in 0:2
        println(io, 1 + x + 3y, " ", 0.04x, " ", 0.04y, " 0.0")
    end
    print(io, "\$EndNodes\n\$Elements\n8\n")
    face = 0
    for y in 0:1, x in 0:1
        a = 1 + x + 3y
        for vertices in ((a, a + 1, a + 4), (a, a + 4, a + 3))
            face += 1
            println(io, face, " 2 2 2 2 ", join(vertices, " "))
        end
    end
    print(io, "\$EndElements\n")
    return String(take!(io))
end


@compile_workload begin
    # Solve one frequency on the CPU backend. Running a whole request is the
    # only way to reach the driver's real call graph, and that graph -- not
    # loading the engine -- was the largest term in a cold start.
    #
    # Build the host call graph without launching accelerator kernels.
    # Requests use the same JSON-decoded type as the worker boundary.
    directory = mktempdir()
    try
        mesh = joinpath(directory, "workload.msh")
        write(mesh, WORKLOAD_MESH)
        request = Dict{String,Any}(
            "schema_version" => 2,
            "beat_engine_backend" => "cpu",
            "frequencies_hz" => [1000.0],
            "config" => Dict{String,Any}(
                "mesh_file" => mesh,
                "scale_factor" => 1.0,
                "distance" => 1.0,
                "axial_offset" => 0.0,
                "step_size" => 90.0,
                "min_angle" => 0.0,
                "max_angle" => 90.0,
                "freq_min" => 1000.0,
                "freq_max" => 1000.0,
                "freq_count" => 1,
                "tag_throat" => 2,
                "rho" => 1.2041,
                "sound_speed" => 343.0,
                "symmetry" => "off",
                "source_motion" => "normal",
            ),
        )
        redirect_stdout(devnull) do
            try
                solve_request(JSON.parse(JSON.json(request)))
                image_mesh = joinpath(directory, "workload_xy.msh")
                write(image_mesh, workload_plate_mesh())
                representative = deepcopy(request)
                representative["frequencies_hz"] = [1000.0, 20000.0]
                representative["config"] = merge(representative["config"], Dict{String,Any}(
                    "mesh_file" => image_mesh, "symmetry" => "xy", "singular_order" => 4,
                    "surface_traces_enabled" => true,
                    "regular_quadrature_mode" => "fixed", "diagonal_enabled" => true,
                    "step_size" => 5.0, "max_angle" => 180.0,
                    "spherical_grid" => Dict("theta_count" => 37, "phi_count" => 72,
                                             "theta_max_deg" => 180.0),
                    "spherical_sampling_enabled" => true,
                    "spherical_sampling_points" => 37 * 72,
                ))
                solve_request(JSON.parse(JSON.json(representative)))
            catch exception
                @warn "BEAT host precompile workload failed" exception=(exception, catch_backtrace())
                # A workload that cannot solve still leaves everything it did
                # reach compiled, and a build must not fail over an
                # optimisation.
            end
        end
    finally
        rm(directory; force=true, recursive=true)
    end

    # The worker's own entry path never runs here -- it reads stdin -- so ask
    # for it by signature. Compiling `worker_loop` is what forces inference
    # through the dynamic `solve_request` call it makes.
    precompile(worker_loop, ())
    precompile(main, (Vector{String},))
end

end
