# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abe07e0) state=d464ee6c sum of radii=1.942146 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def expand_circles(centers, radii, max_iter=2000):
    """
    Expands circles iteratively and resolves overlaps to find a dense packing.
    Uses a repulsion force model to push circles apart when they overlap.
    """
    n = len(radii)
    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)
    
    growth_rate = 1.005
    
    for step in range(max_iter):
        # Try to expand radii multiplicatively
        radii *= growth_rate
        
        dx = np.zeros(n)
        dy = np.zeros(n)
        
        # 1. Resolve boundary violations
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # If circle extends beyond left boundary (x - r < 0)
            if x < r:
                dx[i] += (r - x)
            # If circle extends beyond right boundary (x + r > 1)
            elif x + r > 1:
                dx[i] -= (x + r - 1)
            
            # If circle extends beyond bottom boundary (y - r < 0)
            if y < r:
                dy[i] += (r - y)
            # If circle extends beyond top boundary (y + r > 1)
            elif y + r > 1:
                dy[i] -= (y + r - 1)
        
        # 2. Resolve overlaps between circles
        for i in range(n):
            for j in range(i + 1, n):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                dist_vec = c1 - c2
                dist_sq = dist_vec[0]**2 + dist_vec[1]**2
                dist = math.sqrt(dist_sq)
                
                min_dist = r1 + r2
                # Check for overlap
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Normal vector pointing from j to i
                    nx = dist_vec[0] / dist
                    ny = dist_vec[1] / dist
                    
                    # Push circles apart equally
                    push = overlap / 2.0
                    dx[i] += nx * push
                    dy[i] += ny * push
                    dx[j] -= nx * push
                    dy[j] -= ny * push
        
        # Apply displacements
        centers[:, 0] += dx
        centers[:, 1] += dy
        
        # 3. Check if the configuration is valid
        valid = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r - 1e-9 or x + r > 1 + 1e-9 or y < r - 1e-9 or y + r > 1 + 1e-9:
                valid = False
                break
        
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        # 4. Adjust growth rate based on validity
        if not valid:
            # Overlaps persist, reduce radii
            radii /= 1.01
            growth_rate *= 0.95
        else:
            # Valid, can try to grow faster (capped)
            growth_rate = min(growth_rate * 1.001, 1.015)

    # Final projection to ensure strict boundary compliance
    for i in range(n):
        r = radii[i]
        x, y = centers[i]
        x = max(r, min(1 - r, x))
        y = max(r, min(1 - r, y))
        centers[i] = [x, y]
        
    # Final repulsion pass to ensure no overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                min_d = radii[i] + radii[j]
                if d < min_d:
                    if d < 1e-12:
                        # Break degeneracy
                        centers[j, 0] += 0.001
                    else:
                        ov = min_d - d
                        nx = (centers[i, 0] - centers[j, 0]) / d
                        ny = (centers[i, 1] - centers[j, 1]) / d
                        centers[i, 0] += nx * ov / 2
                        centers[i, 1] += ny * ov / 2
                        centers[j, 0] -= nx * ov / 2
                        centers[j, 1] -= ny * ov / 2
                        changed = True
        if not changed:
            break
            
    return centers, radii

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Initialization 1: 5x5 Grid + 1 center (Structured)
    coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    c1 = np.array([[x, y] for x in coords for y in coords])
    c1 = np.vstack([c1, [0.2, 0.2]])
    r1 = np.full(n, 0.005)
    
    # Initialization 2: Random (Exploration)
    np.random.seed(42)
    c2 = np.random.rand(n, 2)
    r2 = np.full(n, 0.005)
    
    # Initialization 3: Corners + Random (Boundary exploitation)
    c3 = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]])
    # Place remaining circles in the central region to avoid immediate overlap with corners
    c3_rest = np.random.rand(n - 4, 2) * 0.5 + 0.25
    c3 = np.vstack([c3, c3_rest])
    r3 = np.full(n, 0.005)
    
    inits = [(c1, r1), (c2, r2), (c3, r3)]
    
    for c_init, r_init in inits:
        centers_sol, radii_sol = expand_circles(c_init, r_init, max_iter=3000)
        s = np.sum(radii_sol)
        if s > best_sum:
            best_sum = s
            best_centers = centers_sol.copy()
            best_radii = radii_sol.copy()
            
    return best_centers, best_radii, best_sum
