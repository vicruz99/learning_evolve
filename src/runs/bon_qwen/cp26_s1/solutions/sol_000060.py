# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fcf75c21) state=bbe581dc sum of radii=2.288365 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    np.random.seed(42)
    n = 26
    best_r = 0.01
    best_centers = None
    
    # Run multiple restarts to avoid poor local optima
    for restart in range(5):
        # Initialize centers randomly in the unit square
        centers = np.random.rand(n, 2)
        r = 0.05
        lr = 0.003
        beta = 800.0  # Penalty weight for forces
        
        for step in range(4000):
            forces = np.zeros((n, 2))
            energy = 0.0
            
            # Boundary constraints and forces
            for i in range(n):
                x, y = centers[i]
                # Left boundary
                if x < r:
                    v = r - x
                    forces[i, 0] += v * beta
                    energy += v**2
                # Right boundary
                if x > 1 - r:
                    v = x - (1 - r)
                    forces[i, 0] -= v * beta
                    energy += v**2
                # Bottom boundary
                if y < r:
                    v = r - y
                    forces[i, 1] += v * beta
                    energy += v**2
                # Top boundary
                if y > 1 - r:
                    v = y - (1 - r)
                    forces[i, 1] -= v * beta
                    energy += v**2
                    
            # Inter-circle constraints and forces
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.hypot(dx, dy)
                    if dist < 2 * r:
                        overlap = 2 * r - dist
                        energy += overlap**2
                        if dist > 1e-9:
                            # Repulsive force proportional to overlap
                            fx = (dx / dist) * overlap * beta
                            fy = (dy / dist) * overlap * beta
                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy
                            
            # Update positions
            centers += forces * lr
            centers = np.clip(centers, 0.0, 1.0)
            
            # Adaptive radius inflation
            if energy < 1e-4:
                r += 0.00025
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                r -= 0.00008
                if r < 0.01: 
                    r = 0.01
                
            # Decay learning rate gradually
            if step % 300 == 0:
                lr *= 0.96
                
    # Ensure we have a valid configuration
    if best_centers is None:
        best_centers = centers
        best_r = 0.05
        
    # Final strict validation adjustment to guarantee 1e-12 tolerance
    min_gap = 1.0
    for i in range(n):
        x, y = best_centers[i]
        min_gap = min(min_gap, x - best_r, 1 - x - best_r, y - best_r, 1 - y - best_r)
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            min_gap = min(min_gap, dist - 2*best_r)
            
    if min_gap < 0:
        best_r += min_gap
        if best_r < 0.001: 
            best_r = 0.001
            
    radii = np.full(n, best_r)
    return best_centers, radii, np.sum(radii)
