# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 07a91dcd) state=6d2ba3bf sum of radii=1.293821 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_repulsion(centers, r, n):
    """Compute repulsive forces between overlapping circles."""
    forces = np.zeros_like(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.hypot(dx, dy)
            if dist < 2 * r and dist > 1e-8:
                k = (2 * r - dist) / dist
                fx, fy = k * dx, k * dy
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
    return forces

def apply_wall_constraints(centers, r):
    """Compute repulsive forces from square boundaries."""
    forces = np.zeros_like(centers)
    margin = r
    for i in range(centers.shape[0]):
        x, y = centers[i]
        if x < margin:
            forces[i, 0] += (margin - x) * 10.0
        if x > 1.0 - margin:
            forces[i, 0] -= (x - (1.0 - margin)) * 10.0
        if y < margin:
            forces[i, 1] += (margin - y) * 10.0
        if y > 1.0 - margin:
            forces[i, 1] -= (y - (1.0 - margin)) * 10.0
    return forces

def simulate_packing(centers, r, steps=3000, decay=0.995):
    """Run force-directed simulation to resolve overlaps for a given radius."""
    n = centers.shape[0]
    c = centers.copy()
    alpha = 0.05
    for _ in range(steps):
        f = compute_repulsion(c, r, n)
        f += apply_wall_constraints(c, r)
        c += alpha * f
        alpha *= decay
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # Initial configuration: 5x5 grid + 1 center
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.19, 0.1 + j * 0.19]
            idx += 1
    centers[idx] = [0.5, 0.5]
    centers += np.random.randn(*centers.shape) * 0.005
    
    # Target radius based on problem statement (2.636 / 26)
    r_target = 0.10138
    
    # Continuation: gradually increase radius to avoid local minima
    current_r = 0.08
    for step_r in np.linspace(current_r, r_target, 15):
        centers = simulate_packing(centers, step_r, steps=3000)
        
    # Compute strictly feasible radius from final configuration
    min_dist = 2.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < min_dist:
                min_dist = d
        for coord in centers[i]:
            if coord < min_dist:
                min_dist = coord
            if 1.0 - coord < min_dist:
                min_dist = 1.0 - coord
                
    # Guarantee validity with a tiny safety margin for floating point comparisons
    r_actual = (min_dist / 2.0) - 1e-9
    radii = np.full(n, r_actual)
    
    return centers, radii, float(np.sum(radii))
