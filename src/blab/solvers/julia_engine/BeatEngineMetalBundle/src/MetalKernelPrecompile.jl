using Metal: MtlDeviceArray, MtlDeviceMatrix, MtlDeviceVector

# Keep signature evaluation inside the workload: internal GPUArrays function
# names can change with dependency upgrades. A stale entry must warn visibly,
# rather than preventing the worker from loading its host-code bundle.
metal_kernel_signatures() = include(joinpath(@__DIR__, "MetalKernelSignatures.jl"))

function clear_metal_precompile_state!()
    # Matches Metal 1.11.1's own precompile cleanup. Objective-C objects are
    # process-local; only GPUCompiler's portable code cache belongs in the image.
    Base.@lock Metal.compiler_configs_lock empty!(Metal._compiler_configs)
    empty!(Metal.kernel_instances)
    Base.@lock Metal.global_queues_lock empty!(Metal.global_queues)
    Base.@lock Metal.batched_queues_lock empty!(Metal.batched_queues)
    empty!(Metal.queue_residency_sets)
    Base.@lock Metal.memory_pressure_stats_lock empty!(Metal._memory_pressure_stats)
    empty!(Metal.device_malloc_bufs)
    empty!(Metal.MTL.last_committed_per_queue)
    empty!(Metal.MTL.submission_state_per_queue)
    empty!(Metal.device_exception_info)
    Metal.reset_binary_archives!()
end

function precompile_metal_kernel_signatures()
    # Metal.functional() is false while generating a package image, even on a
    # GPU host. Follow Metal's platform gate; device failures warn below.
    if !(Sys.isapple() && Sys.ARCH === :aarch64)
        @info "BEAT Metal kernel workload skipped: requires Apple Silicon"
        return
    end
    # Before Julia 1.11 device inference over shared Base methods can be
    # serialised into the image and break host code later; Metal.jl gates its
    # own real-kernel workload the same way.
    if VERSION < v"1.11"
        @info "BEAT Metal kernel workload skipped: requires Julia 1.11 or newer" VERSION
        return
    end
    successes = 0
    failures = 0
    try
        signatures = metal_kernel_signatures()
        isempty(signatures) && error("No production Metal kernel signatures.")
        for (index, (f, tt)) in enumerate(signatures)
            try
                # Compile and link only; never submit a GPU command buffer.
                Metal.mtlfunction(f, tt)
                successes += 1
            catch exception
                failures += 1
                @warn "BEAT Metal kernel precompile failed" index function_type=typeof(f) argument_types=tt exception=(exception, catch_backtrace())
            end
        end
    catch exception
        failures += 1
        @warn "BEAT Metal kernel inventory could not be loaded" exception=(exception, catch_backtrace())
    finally
        clear_metal_precompile_state!()
        @info "BEAT Metal kernel workload complete" compiled_signatures=successes failures
    end
end
