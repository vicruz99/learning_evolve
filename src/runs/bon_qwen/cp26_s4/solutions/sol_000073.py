# sol_000073 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05693c56) state=e66668f8 sum of radii=0.671798 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(v[2*n:])

def boundary_constraint_fun(v, i, n):
    """Boundary constraints for circle i: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0"""
    r = v[2*n + i]
    return np.array([v[i] - r, 1.0 - v[i] - r, v[n + i] - r, 1.0 - v[n + i] - r])

def pair_constraint_fun(v, i, j, n):
    """Non-overlap constraint for circles i and j: dist^2 - (ri + rj)^2 >= 0"""
    return (v[i] - v[j])**2 + (v[n + i] - v[n + j])**2 - (v[2*n + i] + v[2*n + j])**2

def run_packing():
    n = 26
    
    # 1. Initialize centers in a hexagonal-like grid arrangement
    centers = np.zeros((n, 2))
    idx = 0
    row_counts = [6, 5, 6, 5, 4]  # 26 circles total
    
    for row in range(5):
        count = row_counts[row]
        y = 0.12 + row * 0.175
        for col in range(count):
            # Hexagonal offset for odd rows
            offset = 0.0875 if row % 2 == 1 else 0.0
            x = 0.12 + col * 0.175 + offset
            centers[idx] = [x, y]
            idx += 1
            
    # 2. Compute feasible initial radii (strictly inside constraints)
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(n):
            if i != j:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                max_r = min(max_r, dist / 2.0)
        radii[i] = max_r * 0.9  # Start at 90% capacity
        
    # Flatten to optimization vector: [x1...x26, y1...y26, r1...r26]
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    # Constraints
    constraints = []
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': boundary_constraint_fun,
            'args': (i, n)
        })
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': pair_constraint_fun,
                'args': (i, j, n)
            })
            
    # 3. Run optimization
    res = minimize(
        objective_func,
        x0,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
    )
    
    best_v = res.x
    best_centers = best_v[:2*n].reshape((n, 2))
    best_radii = best_v[2*n:]
    
    # 4. Post-processing: strictly enforce feasibility and extract true max radii
    # This handles any minor numerical drift and guarantees validate_packing passes
    final_radii = np.zeros(n)
    for i in range(n):
        x, y = best_centers[i]
        # Max radius allowed by boundaries
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        
        # Max radius allowed by other circles
        for j in range(n):
            if i != j:
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                # dist >= ri + rj  =>  ri <= dist - rj
                max_r = min(max_r, dist - best_radii[j])
                
        # Apply strict feasibility margin
        final_radii[i] = max(0.0, max_r - 1e-8)
        
    final_sum = np.sum(final_radii)
    return best_centers, final_radii, final_sum
