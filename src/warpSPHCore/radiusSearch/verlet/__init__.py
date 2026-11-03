from ..._backend import BACKEND

if BACKEND == "native":
    from ..native.nt_verlet import (buildVerletList, filterVerletList, verletStaleFlag,
                                    verletUsage, deferVerletChecks)
    __all__ = ["buildVerletList", "filterVerletList", "verletStaleFlag", "verletUsage", "deferVerletChecks"]
else:
    from .build import buildVerletList, deferVerletChecks, verletStaleFlag, verletUsage
    from .filter import filterVerletList, updateNeighborsVerlet

    __all__ = [
        "buildVerletList",
        "verletStaleFlag",
        "verletUsage",
        "deferVerletChecks",
        "filterVerletList",
        "updateNeighborsVerlet"
    ]
