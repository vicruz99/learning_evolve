# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f294fc76) state=d23bc028 sum of radii=2.603349 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# --- Helper Functions ---

def build_constraints(centers_init):
    """
    Construct a list of constraint dictionaries for scipy.optimize.minimize.
    Variables vector x is structured as [x1, y1, r1, x2, y2, r2, ...]
    """
    n = centers_init.shape[0]
    constraints = []
    
    # Boundary Constraints: r <= x <= 1-r  =>  x - r >= 0  and  1 - x - r >= 0
    for i in range(n):
        # x - r >= 0
        def ineq_x_lower(v, idx=i):
            return v[3*idx] - v[3*idx+2]
        constraints.append({'type': 'ineq', 'fun': ineq_x_lower})

        # 1 - x - r >= 0
        def ineq_x_upper(v, idx=i):
            return 1.0 - v[3*idx] - v[3*idx+2]
        constraints.append({'type': 'ineq', 'fun': ineq_x_upper})

        # y - r >= 0
        def ineq_y_lower(v, idx=i):
            return v[3*idx+1] - v[3*idx+2]
        constraints.append({'type': 'ineq', 'fun': ineq_y_lower})

        # 1 - y - r >= 0
        def ineq_y_upper(v, idx=i):
            return 1.0 - v[3*idx+1] - v[3*idx+2]
        constraints.append({'type': 'ineq', 'fun': ineq_y_upper})

    # Non-overlap Constraints: dist_sq >= (r1 + r2)^2
    for i in range(n):
        for j in range(i + 1, n):
            def ineq_overlap(v, idx1=i, idx2=j):
                # Extract coordinates and radii
                x1, y1, r1 = v[3*idx1], v[3*idx1+1], v[3*idx1+2]
                x2, y2, r2 = v[3*idx2], v[3*idx2+1], v[3*idx2+2]
                
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                min_dist = r1 + r2
                min_dist_sq = min_dist * min_dist
                
                return dist_sq - min_dist_sq
            constraints.append({'type': 'ineq', 'fun': ineq_overlap})
            
    return constraints

def generate_initial_config():
    """
    Generates an initial feasible configuration of 26 circles using a hexagonal pattern.
    """
    n = 26
    # We want to place 26 circles. A hexagonal pattern is denser.
    # Row counts: 6, 5, 6, 5, 4 sums to 26.
    row_counts = [6, 5, 6, 5, 4]
    
    centers = []
    radii = []
    
    # Heuristic for vertical spacing based on hexagonal packing (sqrt(3)/2 * diameter)
    # If r is approx 0.1, diameter 0.2. Row spacing ~0.1732.
    # 5 rows: 2*r + 4*spacing. 
    # Let's set a safe initial radius and position them.
    
    # Estimated bounds to fit 5 rows roughly
    # Height needed ~ 1.0. 
    # Let's distribute y coordinates evenly.
    
    y_step = 0.9 / 4.0 # Space 5 rows from 0.1 to 0.9
    r_init = 0.06
    
    current_y = 0.1
    
    for count in row_counts:
        # x coordinates
        # Width available 0.8 (from 0.1 to 0.9)
        # Spacing roughly 0.2 for r=0.1, so 0.12 for r=0.06
        if count > 0:
            x_start = 0.1
            if count > 1:
                x_step = 0.8 / (count - 1)
            else:
                x_step = 0
            
            for k in range(count):
                x = x_start + k * x_step
                centers.append([x, current_y])
                radii.append(r_init)
        
        current_y += y_step
        
    # Hexagonal shift for odd rows to simulate hex lattice
    # Rows 0, 2, 4 are aligned. Rows 1, 3 are shifted.
    # Shift amount should be half the x_step approx.
    # Actually, just a simple shift helps break symmetry.
    
    centers = np.array(centers)
    # Shift rows 1 and 3 slightly to the right
    row_idx = 0
    shift = 0.04 
    for i, count in enumerate(row_counts):
        if i % 2 == 1: # Shift these rows
            start_idx = sum(row_counts[:i])
            end_idx = start_idx + count
            centers[start_idx:end_idx, 0] += shift
            
    # Ensure boundaries are respected for initial radii
    for i in range(n):
        c = centers[i]
        r = radii[i]
        centers[i, 0] = np.clip(c[0], r, 1 - r)
        centers[i, 1] = np.clip(c[1], r, 1 - r)

    return centers, np.array(radii)

def run_packing():
    """
    Run the packing optimization to find centers, radii, and sum of radii.
    """
    try:
        centers_init, radii_init = generate_initial_config()
        n = centers_init.shape[0]
        
        # Combine into a single optimization vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
        
        # Bounds for variables
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])
            
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        def objective(v):
            sum_r = 0.0
            for i in range(n):
                sum_r += v[3*i + 2]
            return -sum_r
            
        cons = build_constraints(centers_init)
        
        # Run optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 200, 'ftol': 1e-8})
        
        # Extract results
        final_centers = np.zeros((n, 2))
        final_radii = np.zeros(n)
        
        for i in range(n):
            final_centers[i, 0] = res.x[3*i]
            final_centers[i, 1] = res.x[3*i+1]
            final_radii[i] = res.x[3*i+2]
            
        # Safety shrink to avoid boundary/overlap issues due to precision
        # We reduce radius slightly to ensure strict compliance with 1e-12 tolerance
        shrink_factor = 0.999
        final_radii *= shrink_factor
        
        # Re-center if needed (though shrinking radii keeps centers valid)
        # Actually, shrinking radii makes the circles smaller, so centers stay valid.
        
        sum_radii = np.sum(final_radii)
        
        return final_centers, final_radii, float(sum_radii)

    except Exception as e:
        # Fallback solution: 5x5 grid plus 1 small circle
        print(f"Optimization failed: {e}")
        centers = []
        radii = []
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
                radii.append(0.1)
        
        # Add 26th circle in the center gap (perturbed)
        # Actually 5x5 center is (0.5, 0.5).
        # We can try to fit a small one, but 25 circles of 0.1 fill the square.
        # Let's just return 25 circles of 0.09 and 1 of 0.09?
        # No, must be 26.
        # Let's return a dense grid with smaller radius.
        # r = 0.08 -> sum = 2.08.
        centers = []
        radii = []
        r = 0.09
        # 5 rows, 5 cols = 25
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
                radii.append(r)
        # 26th circle
        centers.append([0.5, 0.5])
        radii.append(r * 0.5) # Smaller
        
        # Wait, (0.5, 0.5) overlaps with center circle of grid.
        # Let's just scale everything down to fit 26 in a grid.
        # 26 circles. sqrt(26) ~ 5.1.
        # Grid 6x5?
        # 6 cols, 5 rows.
        # Width 1/6 ~ 0.166. Radius ~ 0.083.
        # 26 * 0.083 = 2.158.
        
        # Better fallback: Hexagonal packing logic from init but with fixed valid r
        centers, radii = generate_initial_config()
        # Ensure validity
        for i in range(len(radii)):
            radii[i] = 0.05 # Safe small radius
            centers[i, 0] = np.clip(centers[i, 0], 0.05, 0.95)
            centers[i, 1] = np.clip(centers[i, 1], 0.05, 0.95)
            
        return centers, radii, float(np.sum(radii))
