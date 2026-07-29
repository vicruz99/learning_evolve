# sol_000337 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9bf69ab6) state=da9645ce sum of radii=1.040000 correctness=1.0
# stdout(first 200): Optimization failed for seed 0: cannot reshape array of size 26 into shape (26,2)
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    num_vars = 3 * N
    
    # Helper to generate initial guess
    def get_initial_guess():
        centers = []
        radii = []
        
        # Try to create a hexagonal-like pattern
        # We want to place 26 points.
        # A 5x5 grid has 25 points. We can add 1.
        # Or use rows of alternating lengths.
        
        # Let's try a 6x5 grid (30 points) and pick a subset, or just dense grid.
        # Actually, let's just use a dense grid of 26 points.
        # 6 columns, 5 rows -> 30 points.
        # Let's just generate points on a grid and take 26.
        
        pts = []
        # Grid spacing
        cols = 6
        rows = 5
        
        for r in range(rows):
            for c in range(cols):
                # Center points slightly inside to allow some radius
                x = (c + 0.5) / cols
                y = (r + 0.5) / rows
                pts.append([x, y])
        
        # We have 30 points. Shuffle and take 26.
        np.random.seed(42)
        indices = np.random.choice(len(pts), N, replace=False)
        selected_centers = np.array([pts[i] for i in indices])
        
        # Initial radii: small enough to not overlap
        # Min distance in grid is approx 1/6 ~ 0.16. 
        # r=0.05 gives sum 0.1 < 0.16. Safe.
        r_init = 0.04
        return selected_centers, np.full(N, r_init)

    # Objective function: maximize sum of radii -> minimize negative sum
    def objective(vars):
        # vars layout: x0, y0, r0, x1, y1, r1, ...
        # Radii are at indices 2, 5, 8, ...
        radii = vars[2::3]
        return -np.sum(radii)

    # Prepare constraints
    # We will define them as a list of dictionaries
    cons = []
    
    # Pre-calculate indices for faster access
    idx_x = np.arange(0, num_vars, 3)
    idx_y = np.arange(1, num_vars, 3)
    idx_r = np.arange(2, num_vars, 3)
    
    # Boundary constraints
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    for i in range(N):
        ix, iy, ir = idx_x[i], idx_y[i], idx_r[i]
        
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, ix=ix, ir=ir: v[ix] - v[ir],
            'jac': lambda v, ix=ix, ir=ir: np.zeros(num_vars) # Gradient: 1 at ix, -1 at ir
        })
        # Fixing jacobian manually for boundary constraints
        def jac_x_r(v, ix, ir):
            grad = np.zeros(num_vars)
            grad[ix] = 1.0
            grad[ir] = -1.0
            return grad
        cons[-1]['jac'] = lambda v, ix=ix, ir=ir: jac_x_r(v, ix, ir)

        # 1 - x - r >= 0
        def fun_1_xr(v, ix, ir): return 1.0 - v[ix] - v[ir]
        def jac_1_xr(v, ix, ir):
            grad = np.zeros(num_vars)
            grad[ix] = -1.0
            grad[ir] = -1.0
            return grad
        cons.append({'type': 'ineq', 'fun': lambda v, ix=ix, ir=ir: fun_1_xr(v, ix, ir), 
                     'jac': lambda v, ix=ix, ir=ir: jac_1_xr(v, ix, ir)})

        # y - r >= 0
        def fun_yr(v, iy, ir): return v[iy] - v[ir]
        def jac_yr(v, iy, ir):
            grad = np.zeros(num_vars)
            grad[iy] = 1.0
            grad[ir] = -1.0
            return grad
        cons.append({'type': 'ineq', 'fun': lambda v, iy=iy, ir=ir: fun_yr(v, iy, ir),
                     'jac': lambda v, iy=iy, ir=ir: jac_yr(v, iy, ir)})

        # 1 - y - r >= 0
        def fun_1_yr(v, iy, ir): return 1.0 - v[iy] - v[ir]
        def jac_1_yr(v, iy, ir):
            grad = np.zeros(num_vars)
            grad[iy] = -1.0
            grad[ir] = -1.0
            return grad
        cons.append({'type': 'ineq', 'fun': lambda v, iy=iy, ir=ir: fun_1_yr(v, iy, ir),
                     'jac': lambda v, iy=iy, ir=ir: jac_1_yr(v, iy, ir)})

    # Pairwise constraints: dist(i, j) - (r_i + r_j) >= 0
    # Indices for i: 3i, 3i+1, 3i+2
    # Indices for j: 3j, 3j+1, 3j+2
    
    for i in range(N):
        for j in range(i + 1, N):
            ix_i, iy_i, ir_i = 3*i, 3*i+1, 3*i+2
            ix_j, iy_j, ir_j = 3*j, 3*j+1, 3*j+2
            
            def fun_pair(v, ii=i, jj=j):
                xi, yi, ri = v[ix_i], v[iy_i], v[ir_i]
                xj, yj, rj = v[ix_j], v[iy_j], v[ir_j]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx*dx + dy*dy)
                return dist - (ri + rj)
            
            def jac_pair(v, ii=i, jj=j):
                grad = np.zeros(num_vars)
                xi, yi, ri = v[ix_i], v[iy_i], v[ir_i]
                xj, yj, rj = v[ix_j], v[iy_j], v[ir_j]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-12:
                    # Gradient undefined or zero, return zeros to avoid error
                    return grad
                
                d_dist_dx = dx / dist
                d_dist_dy = dy / dist
                
                # Gradient w.r.t xi, yi, ri
                grad[ix_i] = d_dist_dx
                grad[iy_i] = d_dist_dy
                grad[ir_i] = -1.0
                
                # Gradient w.r.t xj, yj, rj
                grad[ix_j] = -d_dist_dx
                grad[iy_j] = -d_dist_dy
                grad[ir_j] = -1.0
                
                return grad
            
            cons.append({
                'type': 'ineq',
                'fun': fun_pair,
                'jac': jac_pair
            })

    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.extend([
            (0.0, 1.0), # x
            (0.0, 1.0), # y
            (0.0, 0.5)  # r
        ])

    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None

    # Run optimization multiple times with different seeds/initializations
    for seed in range(5):
        np.random.seed(seed + 100)
        centers_init, radii_init = get_initial_guess()
        
        # Flatten initial guess
        x0 = np.zeros(num_vars)
        for i in range(N):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
        
        # Add small random noise to break symmetry and avoid getting stuck
        noise = np.random.normal(0, 0.01, num_vars)
        # Project noise to keep within bounds roughly
        # Just add it, bounds will handle it
        x0 += noise
        
        # Clip to valid range for r just in case
        for i in range(N):
            x0[3*i+2] = max(0.0, x0[3*i+2])
            x0[3*i] = np.clip(x0[3*i], 0.0, 1.0)
            x0[3*i+1] = np.clip(x0[3*i+1], 0.0, 1.0)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'ftol': 1e-8, 'maxiter': 1000})
            
            if res.success or res.fun < 0: # res.fun is negative sum
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = res.x[::3].reshape(N, 2)
                    best_radii = res.x[2::3]
        except Exception as e:
            print(f"Optimization failed for seed {seed}: {e}")

    # Fallback if optimization failed
    if best_centers is None:
        centers_init, radii_init = get_initial_guess()
        best_centers = centers_init
        best_radii = radii_init
        best_sum_radii = np.sum(best_radii)

    # Final validation and cleaning
    # Ensure no NaNs
    if np.isnan(best_centers).any() or np.isnan(best_radii).any():
        print("NaN detected, using fallback")
        centers_init, radii_init = get_initial_guess()
        best_centers = centers_init
        best_radii = radii_init
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii
