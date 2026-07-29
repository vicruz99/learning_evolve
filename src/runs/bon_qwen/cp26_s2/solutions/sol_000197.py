# sol_000197 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bc246a9d) state=c23df6a9 sum of radii=2.569032 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def run_packing():
    """
    Optimizes the positions and radii of 26 circles in a unit square 
    to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialize with a Hexagonal Grid Layout
    # 6 rows, alternating 5 and 4 circles (Total 26)
    # Pattern: 5, 4, 5, 4, 4, 4 -> 26 circles
    # We scale initial radius to fit comfortably
    initial_r = 0.08
    
    centers = []
    row_y = 0
    row_idx = 0
    
    # Row configurations
    # Row 0: 5 circles (x shift 0)
    # Row 1: 4 circles (x shift r)
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 4 circles (to make sum 26: 5+4+5+4+4+4=26)
    # Row 5: 4 circles
    row_counts = [5, 4, 5, 4, 4, 4]
    
    current_y = initial_r # Start at radius from bottom
    h = initial_r * np.sqrt(3) # Hexagonal row height
    
    for i, count in enumerate(row_counts):
        # Shift x for odd rows (staggered)
        x_start = initial_r
        if i % 2 == 1:
            x_start += initial_r 
            
        # Distribute circles evenly in the row
        # Width available approx 1 - 2*r
        # Spacing 2*r
        # We just place them at arithmetic progression
        # To ensure they fit, we center them or pack them.
        # Let's pack them starting from left margin
        
        # Calculate x positions
        # x_k = x_start + k * 2*r
        for k in range(count):
            x = x_start + k * (2 * initial_r)
            centers.append([x, current_y])
            
        current_y += h

    centers = np.array(centers[:n]) # Ensure we have exactly n
    
    # Initial radii
    radii = np.full(n, initial_r)
    
    # 2. Optimization Setup
    # Variables: x1, y1, r1, x2, y2, r2, ...
    x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    
    # Bounds for x, y in [0, 1] and r in [0, 1]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 1)])
        
    # Objective: Minimize -sum(radii)
    def objective(vars):
        r = vars[2::3]
        return -np.sum(r)

    # Constraints
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        # x - r >= 0
        cons = {
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]
        }
        constraints.append(cons)
        
        # 1 - x - r >= 0
        cons = {
            'type': 'ineq',
            'fun': lambda v, idx=i: 1 - v[3*idx] - v[3*idx+2]
        }
        constraints.append(cons)
        
        # y - r >= 0
        cons = {
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]
        }
        constraints.append(cons)
        
        # 1 - y - r >= 0
        cons = {
            'type': 'ineq',
            'fun': lambda v, idx=i: 1 - v[3*idx+1] - v[3*idx+2]
        }
        constraints.append(cons)

    # Non-overlap constraints: dist(i,j) >= ri + rj
    # Vectorized calculation for efficiency
    def non_overlap_constraint(vars):
        # Extract coordinates and radii
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        # Compute pairwise squared distances
        # Using broadcasting or pdist
        # pdist is efficient for (N, 2) array
        pts = np.column_stack((xs, ys))
        dists = pdist(pts) # (N*(N-1)/2, )
        
        # We need indices for radii. pdist doesn't give indices directly in a simple array format for constraint logic
        # Let's do manual loop or vectorized diff
        # For N=26, loop is fine
        
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                # dx, dy
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                r_sum = rs[i] + rs[j]
                
                # Constraint: dist >= r_i + r_j  =>  dist^2 >= (r_i + r_j)^2
                # d_sq - (r_sum)^2 >= 0
                res.append(d_sq - r_sum**2)
                
        return np.array(res)

    # To make it compatible with SLSQP which expects scalar or array for fun
    # We can pass a function that returns the array
    cons_no_overlap = {
        'type': 'ineq',
        'fun': non_overlap_constraint
    }
    constraints.append(cons_no_overlap)

    # 3. Run Optimizer
    # SLSQP is suitable for constrained optimization
    # maxiter and options might need tuning
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    
    # 4. Extract Results
    if res.success:
        x_opt = res.x[0::3]
        y_opt = res.x[1::3]
        r_opt = res.x[2::3]
    else:
        # Fallback to initial if optimization fails (unlikely given valid start)
        x_opt = centers[:, 0]
        y_opt = centers[:, 1]
        r_opt = radii
        
    centers_final = np.column_stack((x_opt, y_opt))
    
    # Ensure radii are non-negative (solver might go slightly negative due to noise, though bounds prevent it)
    r_opt = np.maximum(r_opt, 0)
    
    sum_radii = np.sum(r_opt)
    
    return centers_final, r_opt, sum_radii

# Helper to verify locally if needed (not required for submission but good practice)
# def validate_packing(centers, radii): ... (as provided in prompt)
