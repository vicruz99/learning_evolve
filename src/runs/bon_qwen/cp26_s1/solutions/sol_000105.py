# sol_000105 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 81b841bb) state=48376d65 sum of radii=2.566019 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Helper to flatten variables
    def get_vars(centers, radii):
        return np.concatenate([centers.flatten(), radii])

    def set_vars(var):
        centers = var[:2*n].reshape((n, 2))
        radii = var[2*n:]
        return centers, radii

    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(var):
        centers, radii = set_vars(var)
        return -np.sum(radii)

    # Constraints
    # 1. Boundary constraints: r <= x <= 1-r  =>  x - r >= 0, 1 - x - r >= 0
    # 2. Non-overlap: dist^2 >= (r_i + r_j)^2
    
    def boundary_constraints(var):
        centers, radii = set_vars(var)
        con = []
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            con.append(x - r)      # x >= r
            con.append(1 - x - r)  # x <= 1-r
            con.append(y - r)      # y >= r
            con.append(1 - y - r)  # y <= 1-r
        return con

    def non_overlap_constraints(var):
        centers, radii = set_vars(var)
        con = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                sum_r = radii[i] + radii[j]
                # dist >= sum_r  <=>  dist^2 >= sum_r^2
                # However, dist^2 - sum_r^2 >= 0 is not smooth at 0? 
                # Actually dist >= sum_r is better handled as dist - sum_r >= 0
                # But calculating sqrt is costly? No, it's fine.
                dist = np.sqrt(dist_sq)
                con.append(dist - sum_r)
        return con

    # Initial guess: Hexagonal-like packing
    # 6 rows. Try to fit 4, 5, 4, 5, 4, 4 circles.
    # Rows at y = 0.1, 0.27, 0.44, 0.61, 0.78, 0.9 approx?
    # Let's use a regular grid perturbed or just a grid.
    # A 5x5 grid has 25. We need 26.
    # Let's try a 6x5 grid (30 spots) and remove 4, or just place 26.
    # Let's construct 6 rows.
    
    centers_init = np.zeros((n, 2))
    radii_init = np.full(n, 0.1)
    
    # Configuration: 4, 5, 4, 5, 4, 4
    row_counts = [4, 5, 4, 5, 4, 4]
    current_idx = 0
    
    # Vertical spacing for 6 rows
    # y coords: r, r + h, r + 2h, ...
    # Let's set y manually for init
    y_coords = np.linspace(0.1, 0.9, 6)
    
    for row_idx, count in enumerate(row_counts):
        y = y_coords[row_idx]
        # x coords: spread evenly in [0.1, 0.9]
        if count == 1:
            x_vals = [0.5]
        else:
            x_vals = np.linspace(0.1, 0.9, count)
        
        for x in x_vals:
            centers_init[current_idx, 0] = x
            centers_init[current_idx, 1] = y
            current_idx += 1
            
    # Perturb slightly to avoid symmetry issues if any
    centers_init += np.random.uniform(-0.01, 0.01, centers_init.shape)
    # Clip to valid range for initialization (though optimizer will fix)
    # Actually, let's just ensure radii are small enough for init validity?
    # 0.1 might overlap if too close.
    # In row of 5, dist is 0.2. 0.1+0.1=0.2. Touching.
    # In row of 4, dist is 0.266. 0.1+0.1=0.2. Gap.
    # Vertical dist approx 0.2. 0.1+0.1=0.2. Touching.
    # So 0.1 is a valid start (tight but valid).
    
    # However, with random perturbation, we might violate.
    # Let's reduce initial radius slightly to ensure feasibility
    radii_init[:] = 0.08
    
    initial_var = get_vars(centers_init, radii_init)
    
    # Bounds: x, y in [0, 1], r in [0, 1]
    # Actually tighter bounds: r in [0, 0.5], x,y in [0,1]
    # But boundary constraints handle the rest.
    bounds = [(0, 1)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Prepare constraints for scipy
    # Boundary constraints
    bnd_con = [{'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n + i]} for i in range(n)] # x >= r
    bnd_con += [{'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i] - v[2*n + i]} for i in range(n)] # 1-x >= r
    bnd_con += [{'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*n + i]} for i in range(n)] # y >= r
    bnd_con += [{'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i+1] - v[2*n + i]} for i in range(n)] # 1-y >= r
    
    # Overlap constraints
    overlap_con = []
    for i in range(n):
        for j in range(i + 1, n):
            # dist - (r_i + r_j) >= 0
            def make_con(ii, jj):
                def con(v):
                    xi, yi = v[2*ii], v[2*ii+1]
                    xj, yj = v[2*jj], v[2*jj+1]
                    ri, rj = v[2*n + ii], v[2*n + jj]
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    return dist - (ri + rj)
                return con
            overlap_con.append({'type': 'ineq', 'fun': make_con(i, j)})
            
    all_constraints = bnd_con + overlap_con
    
    # Optimization
    # SLSQP is good for constrained optimization
    result = minimize(objective, initial_var, method='SLSQP', bounds=bounds, constraints=all_constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    
    if result.success or result.fun > -2.0: # Heuristic check
        final_centers, final_radii = set_vars(result.x)
        sum_radii = np.sum(final_radii)
        
        # Post-processing: ensure strict validity (fix small negative violations due to numerical error)
        # But scipy should handle it.
        # Just return
        return final_centers, final_radii, sum_radii
    else:
        # Fallback to the initialization if optimization fails completely
        return centers_init, radii_init, np.sum(radii_init)

# Helper functions to avoid closures as requested
# The closure inside run_packing is unavoidable for the lambda/inner functions in constraints 
# unless we define them globally or use a class. 
# The prompt says "Make all helper functions top level and have no closures from function nesting."
# I will refactor to use top-level functions.

def get_vars_global(centers, radii):
    return np.concatenate([centers.flatten(), radii])

def set_vars_global(var):
    n = var.size // 3 # Wait, var has 2n + n = 3n elements? No, 2 coords + 1 radius = 3 vars per circle.
    # Actually var size is 3*n.
    n = var.size // 3
    centers = var[:2*n].reshape((n, 2))
    radii = var[2*n:]
    return centers, radii

def objective_global(var):
    centers, radii = set_vars_global(var)
    return -np.sum(radii)

def boundary_con_x_ge_r(var, i):
    centers, radii = set_vars_global(var)
    return centers[i, 0] - radii[i]

def boundary_con_1_x_ge_r(var, i):
    centers, radii = set_vars_global(var)
    return 1 - centers[i, 0] - radii[i]

def boundary_con_y_ge_r(var, i):
    centers, radii = set_vars_global(var)
    return centers[i, 1] - radii[i]

def boundary_con_1_y_ge_r(var, i):
    centers, radii = set_vars_global(var)
    return 1 - centers[i, 1] - radii[i]

def overlap_con(var, i, j):
    centers, radii = set_vars_global(var)
    xi, yi = centers[i, 0], centers[i, 1]
    xj, yj = centers[j, 0], centers[j, 1]
    ri, rj = radii[i], radii[j]
    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
    return dist - (ri + rj)

def run_packing():
    n = 26
    
    # Initial guess
    centers_init = np.zeros((n, 2))
    radii_init = np.full(n, 0.08)
    
    # Hexagonal-ish layout: 4, 5, 4, 5, 4, 4
    row_counts = [4, 5, 4, 5, 4, 4]
    current_idx = 0
    y_coords = np.linspace(0.12, 0.88, 6) # Slightly inside
    
    for row_idx, count in enumerate(row_counts):
        y = y_coords[row_idx]
        if count == 1:
            x_vals = [0.5]
        else:
            x_vals = np.linspace(0.1, 0.9, count)
        
        for x in x_vals:
            centers_init[current_idx, 0] = x
            centers_init[current_idx, 1] = y
            current_idx += 1
            
    initial_var = get_vars_global(centers_init, radii_init)
    
    # Bounds
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_con_x_ge_r(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_con_1_x_ge_r(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_con_y_ge_r(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_con_1_y_ge_r(v, i)})
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: overlap_con(v, i, j)})
            
    result = minimize(objective_global, initial_var, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})
    
    final_centers, final_radii = set_vars_global(result.x)
    sum_radii = np.sum(final_radii)
    
    # Clean up any tiny violations due to precision
    # Check and clip radii to 0 if negative
    final_radii = np.maximum(final_radii, 0.0)
    
    # Re-center if slightly out? No, constraints should hold.
    
    return final_centers, final_radii, sum_radii
