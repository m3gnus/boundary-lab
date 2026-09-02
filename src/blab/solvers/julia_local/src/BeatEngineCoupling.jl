# Burton-Miller coupling parameter, and its low-frequency cap.
#
#   eta = i |k| / max(k^2, c)
#
# `c = 0` is the bare eta = i/k this engine has always used, reproduced bit for
# bit (the branch below evaluates `inv(|k|)`, not `|k| / k^2`). A positive `c`
# is the capped, Duhamel-style variant catalogued in the low-frequency
# sound-hard-sphere study, https://arxiv.org/abs/2405.10722, which measures the
# bare i/k degrading conditioning as k goes to zero.
#
# Sign. The engine's time convention is e^{-i omega t} and the coupling is
# +i * (a positive real scale). That is what `Complex{T}(0, 1) / k` was and what
# the capped form keeps: only the real magnitude changes. Nothing here may flip
# it. Marburg measured ~900 GMRES iterations at frequencies where the sign is
# wrong against under 50 where it is right
# (https://www.worldscientific.com/doi/10.1142/S0218396X15500162), and a sign
# error passes every operator-equivalence test, so the cap and the sign are kept
# in one place on purpose.
#
# Choosing c. The coupling multiplies H against (0.5 I - D). Under Galerkin
# testing H is an operator of order +1: on a body of characteristic size R its
# scale is 1/R, while 0.5 I - D is order 0 with scale 1. Balancing the two terms
# therefore wants |eta| ~ R. The bare |eta| = 1/k exceeds R exactly when kR < 1,
# which is the acoustically compact regime -- so the cap is
#
#   c = 1/R^2,   R = half the diagonal of the body's bounding box.
#
# Above kR = 1 the expression returns 1/k unchanged; below it |eta| = k R^2 < R.
# The transition is continuous and its derivative is bounded, so no frequency in
# a sweep sees a jump. On a sphere of radius a the constructed R is a, which is
# the case the cited low-frequency study is written against.
#
# R is derived from the mesh, not tuned. The bounding box is completed across
# whichever symmetry planes are active, because a reduced mesh holds a fraction
# of the body and the coupling has to see the whole of it.

const BEAT_BM_COUPLING_CAP_ENV = "BLAB_BEAT_BM_COUPLING_CAP"

"""
    burton_miller_coupling_scale(k, cap)

The real, positive magnitude of `eta = i * scale`: `|k| / max(k^2, cap)`.

`cap = 0` returns `inv(|k|)` by the same floating-point operation the uncapped
path used, so an uncapped run is unchanged to the last bit.
"""
@inline function burton_miller_coupling_scale(k::T, cap::Real=zero(T)) where {T<:AbstractFloat}
    magnitude = abs(k)
    threshold = T(cap)
    return magnitude * magnitude >= threshold ? inv(magnitude) : magnitude / threshold
end

"""
    burton_miller_coupling(k, cap)

`eta = i |k| / max(k^2, cap)`, the coupling itself.
"""
@inline function burton_miller_coupling(k::T, cap::Real=zero(T)) where {T<:AbstractFloat}
    return Complex{T}(zero(T), burton_miller_coupling_scale(k, cap))
end

"""
    burton_miller_body_radius(mesh; symmetry_mode=:off)

Half the diagonal of the mesh's axis-aligned bounding box, completed across the
active symmetry planes. This is `R` in `c = 1/R^2`.
"""
function burton_miller_body_radius(mesh::BoundaryMesh{T}; symmetry_mode=:off) where {T<:AbstractFloat}
    isempty(mesh.vertices) && error("Cannot derive a Burton-Miller coupling cap from a mesh with no vertices.")
    transforms = symmetry_transforms(symmetry_mode; include_identity=true)
    lower = SVector{3,T}(T(Inf), T(Inf), T(Inf))
    upper = SVector{3,T}(T(-Inf), T(-Inf), T(-Inf))
    for transform in transforms
        for vertex in mesh.vertices
            image = reflect_point(transform, vertex)
            lower = min.(lower, image)
            upper = max.(upper, image)
        end
    end
    extent = upper - lower
    return T(0.5) * sqrt(extent[1]^2 + extent[2]^2 + extent[3]^2)
end

"""
    burton_miller_coupling_cap(mesh; symmetry_mode=:off, override=ENV[...])

`c` for `eta = i |k| / max(k^2, c)`, in inverse square metres.

`override` is read from `$BEAT_BM_COUPLING_CAP_ENV` and accepts:

  * `auto` (the default) -- derive `c = 1/R^2` from the mesh;
  * `off`, `none` or `0` -- no cap, the bare `eta = i/k`;
  * a positive number -- that `c` verbatim, for A/B measurement.

A zero return means an uncapped solve, which is what every caller gets by
default at the function level; only the entry points that know the mesh derive
a cap.
"""
function burton_miller_coupling_cap(
    mesh::BoundaryMesh{T};
    symmetry_mode=:off,
    override::AbstractString=get(ENV, BEAT_BM_COUPLING_CAP_ENV, "auto"),
) where {T<:AbstractFloat}
    text = lowercase(strip(override))
    if text in ("off", "none", "0")
        return zero(T)
    elseif !isempty(text) && text != "auto"
        value = tryparse(Float64, text)
        value === nothing && error(
            "$BEAT_BM_COUPLING_CAP_ENV must be auto, off, or a non-negative number; got $(repr(override)).",
        )
        value >= 0 || error("$BEAT_BM_COUPLING_CAP_ENV must be non-negative; got $value.")
        return T(value)
    end
    radius = burton_miller_body_radius(mesh; symmetry_mode=symmetry_mode)
    radius > zero(T) || error("Cannot derive a Burton-Miller coupling cap from a mesh of zero extent.")
    return inv(radius * radius)
end
