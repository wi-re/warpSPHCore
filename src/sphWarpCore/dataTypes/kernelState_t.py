import warp as wp

@wp.struct
class kernelState:
    kernelFunction: wp.int32
    supportMode: wp.uint32

    gradientMode: wp.int32
    laplacianMode: wp.int32

    positiveDivergenceMode: wp.bool
    divergenceMode: wp.bool

    operationMode: wp.int32