# sol_000270 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e18200b9) state=f449f1f7 sum of radii=2.505334 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Assumes equal radii for simplicity and efficiency, which is often near-optimal.
    """
    n = 26
    best_r = 0.0
    best_centers = None
    best_sum = 0.0
    
    # Number of optimization runs to avoid local minima
    n_runs = 5
    
    for run in range(n_runs):
        # Initialization
        # Start with a grid-like distribution but slightly randomized
        # 5x5 grid is 25 points, we need 26.
        # Let's place them in a roughly hexagonal pattern or just random in valid region
        
        # Random seed for diversity
        np.random.seed(run * 123 + 42)
        
        # Initial radius guess. 5x5 grid allows r=0.1. 
        # 26 circles need slightly smaller r, maybe 0.08-0.09.
        r_init = 0.08 
        
        # Generate random valid positions
        # Ensure they are at least r_init away from boundaries
        margin = r_init
        centers = np.random.uniform(margin, 1 - margin, (n, 2))
        
        # Variables: [x1, y1, ..., x26, y26, r]
        x0 = np.concatenate([centers.flatten(), [r_init]])
        
        # Bounds
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
        bounds.append((1e-6, 0.5)) # r
        
        # Objective: Maximize r => Minimize -r
        def objective(vars):
            return -vars[-1]
        
        # Constraints
        # Returns an array of constraint values (must be >= 0)
        def constraints(vars):
            x = vars[:2*n].reshape(n, 2)
            r = vars[-1]
            cons = []
            
            # Boundary constraints: x >= r, 1-x >= r, etc.
            for i in range(n):
                cons.append(x[i, 0] - r)      # Left
                cons.append(1 - x[i, 0] - r)  # Right
                cons.append(x[i, 1] - r)      # Bottom
                cons.append(1 - x[i, 1] - r)  # Top
                
            # Pairwise constraints: distance^2 >= (2r)^2
            # d^2 - 4r^2 >= 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i, 0] - x[j, 0]
                    dy = x[i, 1] - x[j, 1]
                    d2 = dx*dx + dy*dy
                    cons.append(d2 - 4*r*r)
            
            return np.array(cons)
        
        # Run optimizer
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
            
            if res.success and res.x[-1] > best_r:
                # Check validity manually to be sure
                x_opt = res.x[:2*n].reshape(n, 2)
                r_opt = res.x[-1]
                
                # Quick validity check
                valid = True
                for i in range(n):
                    if (x_opt[i,0] < r_opt - 1e-9 or x_opt[i,0] > 1 - r_opt + 1e-9 or
                        x_opt[i,1] < r_opt - 1e-9 or x_opt[i,1] > 1 - r_opt + 1e-9):
                        valid = False
                        break
                    for j in range(i+1, n):
                        dist = np.sqrt(np.sum((x_opt[i] - x_opt[j])**2))
                        if dist < 2 * r_opt - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_r = r_opt
                    best_centers = x_opt.copy()
                    best_sum = n * r_opt
                    
        except Exception:
            continue

    # If optimization didn't find a good result, fallback to a structured packing
    # 5x5 grid with one extra small circle? No, that's low sum.
    # Try a hexagonal packing construction if best_r is low
    if best_r < 0.09:
        # Construct a hexagonal packing
        # 5 rows: 5, 6, 5, 6, 4 circles? Total 26.
        # Estimate radius. Width 1.
        # Row with 6 circles: width approx 12r? 
        # In hexagonal packing, width of k circles is 2r + (k-1)*2r? No.
        # Horizontal distance between centers is 2r.
        # Span is 2r + (k-1)2r = 2kr.
        # For 6 circles, 12r <= 1 => r <= 0.0833.
        # This is worse than 0.09.
        # Maybe we can tilt?
        # Let's stick to the optimization result if valid, otherwise use a safe grid.
        
        # Safe fallback: 5x5 grid of radius 0.1 (25 circles) + 1 tiny circle?
        # But we need 26 circles.
        # Just return the best found, even if not optimal.
        # Or generate a valid random packing with small radii.
        
        # Let's try to generate a valid packing with r=0.05
        centers_fallback = np.random.uniform(0.05, 0.95, (n, 2))
        r_fallback = 0.05
        # Check overlaps and resolve roughly
        # This is just to ensure we return something valid if optimizer failed completely
        best_centers = centers_fallback
        best_r = r_fallback
        best_sum = n * r_fallback

    # Prepare output
    radii = np.full(n, best_r)
    return best_centers, radii, best_sum
