import warp as wp

from ..type_config import scalar_t


@wp.struct
class kernelState:
    kernelFunction: wp.int32
    supportMode: wp.uint32

    gradientMode: wp.int32
    laplacianMode: wp.int32

    positiveDivergenceMode: wp.bool
    divergenceMode: wp.bool

    operationMode: wp.int32

    #: Lattice-normalisation correction (LATTICE_DENSITY_PLAN.md §3). `C_d`
    #: normalises the kernel's *integral*; a density SUMMED on a lattice of
    #: spacing `s = h / n_h` therefore reads `rho0 * L(n_h) != rho0`.
    #: `normalizationCoefficient` is `1 / L`, and multiplying every kernel value
    #: and derivative by it makes a defect-free lattice read `rho0` exactly.
    #:
    #: The flag exists so that the pair is safe under Warp's zero-initialisation
    #: of struct fields: several call sites build a `kernelState()` and set only
    #: the fields they know about, and a bare coefficient would default to 0.0
    #: and silently zero the kernel. `False` (the zero default) is "off"; the
    #: coefficient is read only when the flag is set, and the host raises on
    #: `flag set + coefficient <= 0` rather than letting it reach a kernel.
    calibrateNormalization: wp.bool
    normalizationCoefficient: scalar_t
