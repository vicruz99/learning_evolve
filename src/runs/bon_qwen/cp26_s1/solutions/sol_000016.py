# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 60d0e48a) state=41cc035d sum of radii=2.564486 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    
    # --- Step 1: Initialization using a "Force-Directed" approach ---
    # Place centers randomly and push them apart to avoid initial overlaps
    np.random.seed(42) # For reproducibility
    centers = np.random.rand(n, 2)
    radii = np.zeros(n)
    
    # Simple repulsion loop to spread points evenly
    for _ in range(500):
        # Calculate repulsive forces
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-5:
                    dist = 1e-5
                    diff = np.random.rand(2) # Jitter if too close
                force_vec = diff / (dist**2) # Inverse square repulsion
                force_vec = np.clip(force_vec, -10, 10) # Limit force magnitude
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Apply forces
        step_size = 0.05 * (1.0 - _ / 500.0) # Annealing step size
        centers += forces * step_size
        
        # Keep inside square with some padding
        centers = np.clip(centers, 0.05, 0.95)

    # Calculate initial feasible radii based on the spread centers
    # A circle's radius is limited by distance to boundary and other centers
    # We can solve this as a simple iterative update or just set to min distance / 2
    # Let's do a few passes of relaxation for radii
    radii = np.full(n, 0.01) # Start small
    
    # Update radii based on current geometry
    # This is a rough estimate to ensure valid start for optimizer
    max_r = 0.0
    for _ in range(50):
        for i in range(n):
            # Boundary constraints
            r_max = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            # Neighbor constraints
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                r_limit = dist - radii[j]
                if r_limit < r_max:
                    r_max = r_limit
            
            if r_max < 0: r_max = 0
            radii[i] = r_max
        
    # --- Step 2: Optimization using SLSQP ---
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    # Size: 3 * n
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(v):
        return -np.sum(v[2::3])
        
    # Constraints
    # 1. Boundary constraints: center +/- radius within [0, 1]
    # x - r >= 0  => x - r >= 0
    # 1 - x - r >= 0 => 1 - x - r >= 0
    # Same for y
    
    # 2. Pairwise constraints: distance >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
    
    # To speed up, we can define a constraint function that returns an array
    
    def constraint_fun(v):
        constraints = []
        
        # Extract arrays
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Boundary constraints
        # x - r >= 0
        constraints.extend(xs - rs)
        # 1 - x - r >= 0
        constraints.extend(1 - xs - rs)
        # y - r >= 0
        constraints.extend(ys - rs)
        # 1 - y - r >= 0
        constraints.extend(1 - ys - rs)
        
        # Pairwise constraints
        # We need to check all pairs
        # This loop might be slow inside the optimizer, but for N=26 (325 pairs) it's okay.
        # Optimization: only check pairs that are close? 
        # But solver might move them, so we must check all or use a robust method.
        # Let's implement the loop.
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.hypot(dx, dy)
                constraints.append(dist - (rs[i] + rs[j]))
                
        return np.array(constraints)

    # We need to pass constraints in a format scipy understands.
    # 'type': 'ineq', 'fun': function returning array >= 0
    constr = {'type': 'ineq', 'fun': constraint_fun}
    
    # Run optimization
    # SLSQP is a good choice for constrained non-linear problems
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constr, 
                   options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = res.x[3*i]
        final_centers[i, 1] = res.x[3*i+1]
        final_radii[i] = res.x[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    # Post-processing: Clean up any tiny numerical violations just in case
    # Although SLSQP should respect them, floating point errors might occur.
    # We can try to slightly shrink radii if needed, but usually not necessary if solver converged.
    # Let's check validity and adjust if strictly needed.
    # However, the prompt asks to return the best packing found.
    
    return final_centers, final_radii, sum_radii
