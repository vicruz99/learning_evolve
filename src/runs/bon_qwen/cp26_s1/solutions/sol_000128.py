# sol_000128 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22de7e34) state=485520e4 sum of radii=2.572024 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # 1. Initialization: 5x5 Grid
    # Placing circles on a regular grid to ensure a valid starting point
    grid_coords = np.linspace(0.1, 0.9, 5)
    x_coords, y_coords = np.meshgrid(grid_coords, grid_coords)
    centers = np.column_stack([x_coords.flatten(), y_coords.flatten()])
    
    # Ensure we have exactly 26 circles (5x5=25, add one slightly offset)
    if centers.shape[0] < n:
        centers = np.vstack([centers, [0.5, 0.5 + 0.2]])
    elif centers.shape[0] > n:
        centers = centers[:n]
        
    # Start with small valid radii
    radii = np.ones(n) * 0.05
    
    # 2. Force-Directed Growth Simulation
    # Grows radii while using repulsion to maintain validity
    dt = 0.01
    repulsion = 500.0
    growth_rate = 0.0005
    
    for _ in range(3000):
        radii += growth_rate
        
        # Vectorized pair-wise distances and radii sums
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Identify overlaps
        overlaps = np.maximum(0, rad_sums - dists)
        
        # Vectorized repulsive forces for overlaps
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Calculate forces: direction * overlap * strength
        force_vectors = unit_vecs * overlaps[:, :, np.newaxis] * repulsion
        
        # Sum forces for each circle
        forces = np.sum(force_vectors, axis=1)
        
        # Boundary forces
        # If x < r, push right (positive force)
        forces[:, 0] += np.maximum(0, radii - centers[:, 0]) * repulsion
        # If x > 1-r, push left (negative force)
        forces[:, 0] -= np.maximum(0, centers[:, 0] - (1 - radii)) * repulsion
        # If y < r, push up
        forces[:, 1] += np.maximum(0, radii - centers[:, 1]) * repulsion
        # If y > 1-r, push down
        forces[:, 1] -= np.maximum(0, centers[:, 1] - (1 - radii)) * repulsion
        
        # Update positions
        centers += forces * dt
        
        # Clamp to [0, 1] to prevent numerical drift, though forces should keep them in
        centers = np.clip(centers, 0, 1)
        
        # Reduce growth rate gradually to settle
        growth_rate *= 0.999

    # 3. Nonlinear Optimization (SLSQP)
    # Refines the packing by maximizing sum(radii) subject to strict constraints
    
    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        return -np.sum(vars[2::3]) # Minimize negative sum of radii

    def constraint_func(vars):
        pts = vars.reshape((n, 3))
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        cons = []
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        cons.extend(x - r)
        cons.extend(1 - x - r)
        cons.extend(y - r)
        cons.extend(1 - y - r)
        
        # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
        # Using squared distance avoids sqrt in constraints, though dist calculation is similar
        # For SLSQP, we check dist >= r_i + r_j
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = diff_x**2 + diff_y**2
        
        rad_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Extract upper triangle to avoid duplicates and self-loops
        tri_indices = np.triu_indices(n, k=1)
        pairwise_cons = dist_sq[tri_indices] - rad_sum[tri_indices]**2
        
        cons.extend(pairwise_cons)
        
        return np.array(cons)

    # Prepare initial guess for optimizer
    x0 = np.zeros(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii

    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    # Constraint specification
    cons = {'type': 'ineq', 'fun': constraint_func}

    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 1000, 'ftol': 1e-10})

    # Extract final results
    final_vars = result.x
    final_centers = np.column_stack([final_vars[0::3], final_vars[1::3]])
    final_radii = final_vars[2::3]
    sum_radii = np.sum(final_radii)

    return final_centers, final_radii, sum_radii
