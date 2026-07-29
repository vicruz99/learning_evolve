# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 608ae89b) state=c2f0325f sum of radii=1.728568 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization and iterative repulsion-based relaxation.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    centers = np.zeros((n, 2))
    idx = 0
    spacing = 0.2  # Initial grid spacing
    for i in range(6):
        for j in range(5):
            if idx >= n: break
            x = (j + 0.5) * spacing + (0.5 * spacing * (i % 2))
            y = (i + 0.5) * spacing * np.sqrt(3) / 2
            centers[idx] = [x, y]
            idx += 1
        if idx >= n: break
        
    # Small random perturbation to break perfect symmetry and avoid local traps
    centers += np.random.randn(n, 2) * 0.005
    
    r = 0.05          # Initial radius
    alpha = 0.06      # Step size / learning rate
    max_iter = 5000   # Total simulation steps
    
    for t in range(max_iter):
        forces = np.zeros_like(centers)
        overlaps = 0
        
        # 2. Pairwise repulsion forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                if dist < 2 * r:
                    if dist > 1e-6:
                        overlap = 2 * r - dist
                        # Force proportional to overlap amount
                        fx = (dx / dist) * overlap * 6.0
                        fy = (dy / dist) * overlap * 6.0
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                        overlaps += 1
                    else:
                        # Prevent division by zero: push apart randomly
                        fx, fy = np.random.randn(2) * 0.1
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                        
        # 3. Boundary repulsion forces
        for i in range(n):
            x, y = centers[i]
            # Left boundary
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            # Right boundary
            elif x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 10.0
                
            # Bottom boundary
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            # Top boundary
            elif y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 10.0
                
        # Update positions with damped step
        centers += alpha * forces
        # Clip to physically valid range for centers
        centers = np.clip(centers, r, 1 - r)
        
        # 4. Adaptive radius growth
        if overlaps == 0:
            # No overlaps: expand circles slightly
            r *= 1.0005
        else:
            # Overlaps exist: reduce step size gradually to settle
            if t > 1000:
                alpha *= 0.995
                
    radii = np.full(n, r)
    return centers, radii, float(np.sum(radii))
