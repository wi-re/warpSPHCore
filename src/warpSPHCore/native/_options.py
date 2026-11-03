"""Internal operator policy encoding shared by primal and derivative bridges."""
from ..enumTypes import WarpOperation

# Stable registrations already carry a Laplacian policy integer. Keep the base
# variant in its low nibble; the independent clipping option uses the next bit.
POSITIVE_DIVERGENCE = 16


def laplacian_policy(properties):
    clipped = properties.operation == WarpOperation.Laplacian and properties.positiveDivergence
    return properties.laplacianMode.value | (POSITIVE_DIVERGENCE if clipped else 0)


def normalization_factor(properties, dimension):
    if not properties.calibrateNormalization:
        return 1.
    import math
    from ..util.latticeDensity import latticeDensityFactor
    factor = latticeDensityFactor(properties.kernel, float(properties.n_h), dimension)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Lattice normalization coefficient must be finite and positive")
    return factor
