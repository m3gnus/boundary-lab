# BEAT Engine Apple Metal backend: shared cache types and runtime controls.
#
# The Metal backend mirrors the ROCm backend file for file. FEM assembly and
# condensation stay on the CPU, the four dense Burton-Miller operators and the
# exterior field are assembled on the GPU, and the dense factorization runs on
# the CPU through LAPACK because Metal.jl provides no GPU LU. Production
# precision is Float32/ComplexF32 like every other BEAT backend, which is also
# the only precision Apple GPUs support.

using Metal: MtlArray, thread_position_in_grid_1d

const BEAT_METAL_OPERATOR_STORAGE_ENV = "BLAB_METAL_OPERATOR_STORAGE"

"""
    metal_operator_storage_mode()

Storage mode for the four dense Burton-Miller operator matrices.

Apple Silicon is a unified-memory device, so a shared-storage buffer can be
wrapped as a host `Array` with no copy at all; a private-storage buffer has to
be blitted through a staging buffer, which measured 3.5-8 GB/s and cost
1.1-1.5 s per frequency at 10,230 P1 dofs. Shared is therefore the default.
Set `BLAB_METAL_OPERATOR_STORAGE=private` to fall back to the copying path.
"""
function metal_operator_storage_mode()
    requested = lowercase(strip(get(ENV, BEAT_METAL_OPERATOR_STORAGE_ENV, "shared")))
    requested in ("shared", "private") ||
        error("$BEAT_METAL_OPERATOR_STORAGE_ENV must be shared or private; got $(repr(requested)).")
    return requested == "shared" ? Metal.SharedStorage : Metal.PrivateStorage
end

"""
    metal_sweep_memory_available()

Bytes the device can still take before it passes Apple's own recommended
working set, read from Metal's counters rather than a tally we keep: the caches,
the resident operators and anything else in the process are already inside
`currentAllocatedSize`. Exceeding the recommendation does not fail an
allocation, it starts paging unified memory, which is why the sweep lookahead
is sized against what is left rather than against the total.
"""
function metal_sweep_memory_available()
    device = Metal.device()
    recommended = Int(device.recommendedMaxWorkingSetSize)
    allocated = Int(device.currentAllocatedSize)
    return max(0, recommended - allocated)
end

struct MetalSingularCorrectionCache{T}
    pair_offsets
    test_indices
    trial_indices
    rule_indices
    jac_scales
    normal_products
    rule_offsets
    rule_test_points
    rule_trial_points
    rule_weights
    pair_count::Int
end

struct MetalRegularAssemblyCache{T,C}
    host_cache::C
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    element_rule_points   # face_count x rule_count x 3: every element's regular quadrature points
    vertex_offsets
    incident_elements
    incident_local_indices
    dp0_elements
    p1_dofs
    element_dp0_dofs
    color_elements
    color_offsets::Vector{Int}
    element_indices::Vector{Int}
    face_count::Int
    p1_dof_count::Int
    dp0_dof_count::Int
    rule_count::Int
    symmetry_mode::Symbol
    image_transforms::Vector{SymmetryTransform}
    image_singular_caches::Vector{MetalSingularCorrectionCache{T}}
    image_singular_pair_count::Int
    gather_tables::Ref{Any}   # MetalGatherTables, built lazily by the pair_gather kernel mode
    fused_gather_tables::Ref{Any}   # MetalFusedGatherTables, built lazily by the fused Burton-Miller path
end

struct MetalFieldEvaluationCache{T}
    source_points
    source_normals
    source_weights
    source_faces
    source_elements
    basis_values
    source_count::Int
end

function _require_metal!()
    Metal.functional() || error("Metal solve requested, but Metal.functional() is false.")
    return nothing
end

function _metal_kernel_groupsize()
    groupsize = parse(Int, get(ENV, "BLAB_METAL_KERNEL_GROUPSIZE", "256"))
    groupsize in (32, 64, 128, 256, 512, 1024) ||
        error("BLAB_METAL_KERNEL_GROUPSIZE must be 32, 64, 128, 256, 512, or 1024; got $(groupsize).")
    return groupsize
end

function _normalized_metal_regular_kernel_mode(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_METAL_REGULAR_KERNEL_MODE", "pair_gather"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :pair => :pair_owned,
        :pair_owned => :pair_owned,
        :colored => :pair_owned,
        :colored_pair_owned => :pair_owned,
        :entry => :entry_owned,
        :entry_owned => :entry_owned,
        :atomic => :pair_atomic,
        :pair_atomic => :pair_atomic,
        :fused_atomic => :pair_atomic,
        :gather => :pair_gather,
        :pair_gather => :pair_gather,
        :chunked => :pair_gather,
        :chunked_pair_gather => :pair_gather,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error(
        "BLAB_METAL_REGULAR_KERNEL_MODE must be pair_gather, pair_atomic, pair_owned, or entry_owned; got $(value).",
    )
    return normalized
end

# Singular corrections can run as device kernels (native) or be computed on the
# CPU and added to the device operators (host). The host mode exists so a kernel
# defect can be separated from a Duffy-rule defect on the same mesh.
function _normalized_metal_singular_mode(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_METAL_SINGULAR_MODE", "native"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :native => :native,
        :device => :native,
        :host => :host,
        :cpu => :host,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error("BLAB_METAL_SINGULAR_MODE must be native or host; got $(value).")
    return normalized
end

@inline function _metal_global_linear_index()
    return Int(thread_position_in_grid_1d())
end

function _metal_launch(kernel, count::Integer, args...; groupsize::Integer=_metal_kernel_groupsize())
    count <= 0 && return nothing
    Metal.@metal threads=groupsize groups=cld(count, groupsize) kernel(args...)
    return nothing
end
