# Native compact-grid host runtime

With WARPSPHCORE_BACKEND=native and WARPSPHCORE_OPERATORS=native, the public
buildCompactHashMap, fixed-grid builders and pure grid geometry do not import
Warp. Cell geometry, stable sorting and hash-bucket layout preserve the reference;
key/hash generation uses the existing C++ fixed_grid_keys operator. Warp runtime
and explicitly selected Warp operators retain their original declarations.

Qualification: exact full descriptor comparisons for 1D/2D/3D, all six support
modes, empty roles, hash collisions and fixed layouts on CPU/CUDA FP32/FP64.
The CUDA probe changes positions and checks actual captured replay against a
fresh eager build. See tests/native/torch/test_host_grid_build.py and
host_grid_probe.py. Retained evidence: frontend-native-runtime-grid-lifecycle-
20261109-r1 in the external experiments workspace. No C++ source or default changed.
