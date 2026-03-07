import warp as wp
@wp.func
def mod_distance(
    x : float, y: float, minDomain: float, maxDomain: float, periodic: bool
):
    if periodic:
        dx = x - y
        domain_size = maxDomain - minDomain
        if wp.abs(dx) > domain_size / 2.0:
            dx = wp.sign(dx) * (wp.abs(dx) - domain_size)
    else:
        dx = x - y
    
    return dx

# @wp.kernel
def computeCartesianDistance_2d(
    x: wp.vec2f,
    y: wp.vec2f,
    minDomain: wp.vec2f,
    maxDomain: wp.vec2f,
    periodic: wp.array(dtype=wp.bool)
):
    dist_sq = float(0.0)
    dx = mod_distance(x[0], y[0], minDomain[0], maxDomain[0], periodic[0])
    dy = mod_distance(x[1], y[1], minDomain[1], maxDomain[1], periodic[1])
    dist_sq = dx * dx + dy * dy
    return wp.sqrt(dist_sq)
    
    
@wp.func
def computeCartesianDistance(
    x: wp.array(dtype=wp.float32),  # Shape (D,)
    y: wp.array(dtype=wp.float32),  # Shape (D,)
    minDomain: wp.array(dtype=wp.float32),  # Shape (D,)
    maxDomain: wp.array(dtype=wp.float32),  # Shape (D,)
    periodic: wp.array(dtype=wp.bool)        # Shape (D,)
):
    dist_sq = float(0.0)
    for d in range(wp.len(x)):
        dx = mod_distance(x[d], y[d], minDomain[d], maxDomain[d], periodic[d])
        dist_sq += dx * dx
    return wp.sqrt(dist_sq)
