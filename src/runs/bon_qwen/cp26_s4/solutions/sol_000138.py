# sol_000138 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d0344893) state=847b5cd2 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Variable indices: 
    # x: 0, 3, 6...
    # y: 1, 4, 7...
    # r: 2, 5, 8...
    
    # Objective: maximize sum(r) => minimize -sum(r)
    def objective(vars_):
        return -np.sum(vars_[2::3])

    # We will define constraints inside the loop to capture n_circles correctly
    # but we need to be careful with efficiency.
    
    for seed in range(5):
        rng = np.random.default_rng(seed)
        
        # Initialization: Dense random placement
        # Start with small radii to ensure feasibility
        r_init = 0.03
        centers = rng.uniform(r_init, 1.0 - r_init, size=(n_circles, 2))
        radii = np.full(n_circles, r_init)
        
        x0 = np.zeros(3 * n_circles)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        
        # Bounds
        bounds = [(0.0, 1.0)] * (3 * n_circles)
        # r bounds
        for k in range(n_circles):
            bounds[3*k + 2] = (0.0, 0.5)
            
        # Constraints
        cons = []
        
        # 1. Boundary constraints (4 per circle)
        for i in range(n_circles):
            idx_x = 3*i
            idx_y = 3*i + 1
            idx_r = 3*i + 2
            
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, ix=idx_x, ir=idx_r: v[ix] - v[ir]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, ix=idx_x, ir=idx_r: 1.0 - v[ix] - v[ir]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, iy=idx_y, ir=idx_r: v[iy] - v[ir]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, iy=idx_y, ir=idx_r: 1.0 - v[iy] - v[ir]})
            
        # 2. Non-overlap constraints (n*(n-1)/2)
        # To speed up, we can skip some or just do all. 325 is manageable.
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                ix, iy, ir = 3*i, 3*i+1, 3*i+2
                jx, jy, jr = 3*j, 3*j+1, 3*j+2
                
                def make_overlap_con(i_x, i_y, i_r, j_x, j_y, j_r):
                    def con(v):
                        dx = v[i_x] - v[j_x]
                        dy = v[i_y] - v[j_y]
                        dr = v[i_r] + v[j_r]
                        return dx*dx + dy*dy - dr*dr
                    return con
                
                cons.append({'type': 'ineq', 'fun': make_overlap_con(ix, iy, ir, jx, jy, jr)})
        
        # Optimize
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if res.success or res.nit > 0:
                current_sum = -res.fun
                # Check validity manually to be sure
                centers_opt = res.x[0::3].reshape(n_circles, 2)
                radii_opt = res.x[2::3]
                
                # Quick validation check
                valid = True
                # Boundary check
                if np.any(radii_opt < 0): valid = False
                if np.any(centers_opt[:, 0] - radii_opt < -1e-6): valid = False
                if np.any(centers_opt[:, 0] + radii_opt > 1 + 1e-6): valid = False
                if np.any(centers_opt[:, 1] - radii_opt < -1e-6): valid = False
                if np.any(centers_opt[:, 1] + radii_opt > 1 + 1e-6): valid = False
                
                # Overlap check
                # Only check if close to violation
                # Using broadcasting for speed? Or just loop
                for i in range(n_circles):
                    for j in range(i+1, n_circles):
                        dist = np.sqrt((centers_opt[i,0]-centers_opt[j,0])**2 + (centers_opt[i,1]-centers_opt[j,1])**2)
                        if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid and current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers_opt
                    best_radii = radii_opt
                    
        except Exception:
            continue

    if best_centers is None:
        # Fallback to a simple grid
        best_centers = np.tile([0.5, 0.5], (n_circles, 1))
        best_radii = np.zeros(n_circles)
        best_sum = 0.0
        
    return best_centers, best_radii, float(best_sum)
