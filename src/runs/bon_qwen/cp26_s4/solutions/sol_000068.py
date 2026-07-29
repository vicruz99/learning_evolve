# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 10bf7585) state=b3663823 sum of radii=2.460107 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def generate_initial_packing(n_circles):
    """
    Generates an initial valid packing of n_circles in a unit square.
    Uses a hexagonal grid pattern.
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Hexagonal packing parameters
    # We want to fit n_circles. 
    # Estimate radius. For N=26, r ~ 0.1. Let's start smaller to be safe.
    r_est = 0.05 
    
    # Row height in hexagonal packing: r * sqrt(3)
    row_height = r_est * math.sqrt(3)
    # Horizontal spacing: 2 * r
    h_spacing = 2 * r_est
    
    count = 0
    row_idx = 0
    
    # Determine how many rows we need
    # Width 1, Height 1.
    # Max cols approx 1 / (2*r)
    # Max rows approx 1 / (r*sqrt(3))
    
    # Let's just place them in rows until we have n_circles
    # Center rows around y=0.5, x=0.5
    
    # Calculate max rows and cols for estimated r
    max_cols = int(1.0 / h_spacing)
    if max_cols < 1: max_cols = 1
    
    # We'll try to fill rows
    current_y = r_est # Start at bottom
    row_number = 0
    
    while count < n_circles:
        # Offset for hexagonal rows
        offset = row_number % 2 * r_est
        
        # Calculate how many circles fit in this row
        # x coords: r + offset, r + offset + 2r, ...
        # Must satisfy x <= 1 - r
        # r + offset + k*2r <= 1 - r  =>  k*2r <= 1 - 2r - offset
        # k <= (1 - 2r - offset) / 2r
        
        if offset > 0:
             # Shifted row
             max_k = int((1.0 - 2*r_est - offset) / h_spacing)
             if max_k < 0: max_k = 0
             # Actually indices 0 to max_k
             num_in_row = max_k + 1
        else:
             # Non-shifted row
             # x starts at r_est
             # r + k*2r <= 1-r => k <= (1-2r)/2r
             max_k = int((1.0 - 2*r_est) / h_spacing)
             if max_k < 0: max_k = 0
             num_in_row = max_k + 1
            
        # Place circles in this row
        for k in range(num_in_row):
            if count >= n_circles:
                break
            x = r_est + offset + k * h_spacing
            y = current_y
            centers[count] = [x, y]
            radii[count] = r_est
            count += 1
            
        current_y += row_height
        row_number += 1
        
    return centers[:n_circles], radii[:n_circles]

def run_packing():
    n = 26
    
    # 1. Generate initial valid packing
    # We try a few different random seeds or patterns to find a good local optimum
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Helper function for constraints
    # We will pass the indices of circles to constraints
    
    # To speed up, we might not check all constraints if n is large, but 26 is small.
    # However, defining 325 constraint functions is verbose.
    # We can define a single constraint function that returns an array of violations?
    # scipy.optimize.minimize constraints can be a dict with 'fun' returning array >= 0.
    
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Index mapping: circle i -> [3*i, 3*i+1, 3*i+2]
    
    def get_coords_and_radii(v):
        # v is 1D array of length 3*n
        # reshape to (n, 3)
        return np.reshape(v, (n, 3))
    
    def objective(v):
        # Maximize sum of radii -> Minimize negative sum
        # radii are at indices 2, 5, 8, ...
        radii = v[2::3]
        return -np.sum(radii)
    
    def constraint_boundary(v):
        # r <= x <= 1-r => x-r >= 0, 1-x-r >= 0
        # r <= y <= 1-r => y-r >= 0, 1-y-r >= 0
        # Returns array of length 4*n
        cr = get_coords_and_radii(v)
        # cr shape (n, 3) -> x, y, r
        x = cr[:, 0]
        y = cr[:, 1]
        r = cr[:, 2]
        
        c1 = x - r
        c2 = (1.0 - x) - r
        c3 = y - r
        c4 = (1.0 - y) - r
        
        return np.concatenate([c1, c2, c3, c4])

    def constraint_non_overlap(v):
        # dist^2 >= (r1 + r2)^2
        # dist^2 - (r1 + r2)^2 >= 0
        cr = get_coords_and_radii(v)
        violations = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = cr[i, 0] - cr[j, 0]
                dy = cr[i, 1] - cr[j, 1]
                dr = cr[i, 2] + cr[j, 2]
                val = dx*dx + dy*dy - dr*dr
                violations.append(val)
        return np.array(violations)

    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5] (actually max r is 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Constraints definitions
    con_bound = {'type': 'ineq', 'fun': constraint_boundary}
    con_overlap = {'type': 'ineq', 'fun': constraint_non_overlap}
    constraints = [con_bound, con_overlap]

    # Initial guess
    # Try a few random variations of a grid/hexagonal packing
    
    # Strategy: Run optimization multiple times with slightly different starts
    best_obj = float('inf')
    best_v = None
    
    # Generate a good starting point
    # Hexagonal grid is good.
    start_centers, start_radii = generate_initial_packing(n)
    
    # Convert to vector
    v0 = np.zeros(3 * n)
    for i in range(n):
        v0[3*i] = start_centers[i, 0]
        v0[3*i+1] = start_centers[i, 1]
        v0[3*i+2] = start_radii[i]
        
    # Try optimizing from this start
    # SLSQP is good for this
    try:
        res = opt.minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success or (res.fun < best_obj):
            best_obj = res.fun
            best_v = res.x
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Try to perturb and re-optimize if needed
    # Since finding global optimum is hard, we rely on the initial good guess.
    # However, the objective is -sum(r). The constraints are active.
    # If the initial radii are small, it should grow.
    
    # Let's check if we can improve by random restarts
    # But 26 circles with 325 constraints is heavy for many restarts.
    # Let's stick to one robust run if possible, or a few.
    
    # Re-run with a slightly perturbed version of the result if found
    if best_v is not None:
        # Perturb
        np.random.seed(42)
        v_pert = best_v + np.random.normal(0, 0.001, size=best_v.shape)
        # Project to bounds
        for i in range(0, 3*n, 3):
            if v_pert[i] < 0: v_pert[i] = 0.0
            if v_pert[i] > 1: v_pert[i] = 1.0
            if v_pert[i+1] < 0: v_pert[i+1] = 0.0
            if v_pert[i+1] > 1: v_pert[i+1] = 1.0
            if v_pert[i+2] < 0: v_pert[i+2] = 0.0
            if v_pert[i+2] > 0.5: v_pert[i+2] = 0.5
            
        try:
            res2 = opt.minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=constraints,
                                options={'maxiter': 1000, 'ftol': 1e-9})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_v = res2.x
        except:
            pass

    # Another start: Random valid packing?
    # Hard to generate random valid packing without overlap.
    # But we can take the best found and try to escape local minima?
    # The problem is non-convex.
    
    # Let's try one more start: Uniform grid but randomized slightly
    rng = np.random.default_rng(123)
    # Just to ensure we don't get stuck in a bad spot, though hex grid is good.
    
    # Final extraction
    if best_v is not None:
        cr = get_coords_and_radii(best_v)
        centers = cr[:, :2]
        radii = cr[:, 2]
    else:
        # Fallback to initial
        centers = start_centers
        radii = start_radii

    sum_radii = np.sum(radii)
    
    # Validate internally (optional but good for debugging)
    # We assume the optimizer respected constraints within tolerance.
    # However, numerical errors might exist.
    # The validation function allows 1e-12 tolerance.
    
    return centers, radii, sum_radii
