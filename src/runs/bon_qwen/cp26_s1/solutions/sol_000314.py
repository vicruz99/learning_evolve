# sol_000314 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a46c309d) state=6a5f62c2 sum of radii=1.640689 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_init_centers():
    centers = []
    # Hexagonal grid parameters
    spacing = 0.13
    offset_x = spacing / 2.0
    
    y = spacing / 2.0
    while y < 1.0:
        row_offset = (int((y - spacing/2.0) / spacing) % 2) * offset_x
        x = spacing / 2.0 + row_offset
        while x < 1.0:
            centers.append([x, y])
            x += spacing
        y += spacing
    
    centers = np.array(centers[:26])
    return centers

def solve_radii(centers):
    n = centers.shape[0]
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Inter-circle constraints
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i +  1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            dists[i, j] = d
            dists[j, i] = d
            
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    # Boundary constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        
        x, y = centers[i]
        A_ub.append(row); b_ub.append(x)
        A_ub.append(row); b_ub.append(1.0 - x)
        A_ub.append(row); b_ub.append(y)
        A_ub.append(row); b_ub.append(1.0 - y)
        
    bounds = [(0, None) for _ in range(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, dists
    return None, dists

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    centers = get_init_centers()
    radii = np.ones(26) * 0.01
    best_sum = 0.0
    best_state = (centers.copy(), radii.copy())
    
    # Optimization parameters
    steps = 800
    step_size = 0.02
    initial_temp = 0.05
    cooling = 0.995
    
    temp = initial_temp
    
    for _ in range(steps):
        step_size *= 0.99
        temp *= cooling
        
        r_opt, dists = solve_radii(centers)
        if r_opt is None:
            continue
            
        current_sum = np.sum(r_opt)
        if current_sum > best_sum:
            best_sum = current_sum
            best_state = (centers.copy(), r_opt.copy())
            
        forces = np.zeros_like(centers)
        
        # Calculate forces based on active constraints
        for i in range(26):
            for j in range(i + 1, 26):
                dist = dists[i, j]
                if dist < 1e-7:
                    dist = 1e-7
                    vec = np.random.rand(2) * 0.01
                else:
                    vec = centers[j] - centers[i]
                    
                # Repulsion if tight
                tightness = (r_opt[i] + r_opt[j]) / dist
                if tightness > 0.95:
                    force_mag = 1.0 / (dist * dist)
                    forces[i] -= (vec / dist) * force_mag
                    forces[j] += (vec / dist) * force_mag
                    
            # Wall forces
            x, y = centers[i]
            if r_opt[i] >= x - 1e-4:
                forces[i, 0] -= 1.0 / (x + 1e-4)
            if r_opt[i] >= 1.0 - x - 1e-4:
                forces[i, 0] += 1.0 / (1.0 - x + 1e-4)
            if r_opt[i] >= y - 1e-4:
                forces[i, 1] -= 1.0 / (y + 1e-4)
            if r_opt[i] >= 1.0 - y - 1e-4:
                forces[i, 1] += 1.0 / (1.0 - y + 1e-4)
                
        # Move centers
        noise = rng.normal(0, temp * 0.01, centers.shape)
        centers += forces * step_size + noise
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
        
    # Final radii calculation
    radii, _ = solve_radii(best_state[0])
    return best_state[0], radii, float(np.sum(radii))
