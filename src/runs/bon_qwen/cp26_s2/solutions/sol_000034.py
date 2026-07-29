# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eaaa636a) state=3c78fc3c sum of radii=2.504236 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(vars):
    """Objective: minimize -r (equivalent to maximizing r)"""
    return -vars[-1]

def compute_constraints(vars):
    """Compute inequality constraints: boundary and non-overlap"""
    r = vars[-1]
    centers = vars[:-1].reshape((N_CIRCLES, 2))
    
    con = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    con.append(centers[:, 0] - r)
    con.append(1.0 - r - centers[:, 0])
    con.append(centers[:, 1] - r)
    con.append(1.0 - r - centers[:, 1])
    
    # Non-overlap constraints: ||c_i - c_j||^2 >= 4r^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    con.append(dist_sq[mask] - 4.0 * r**2)
    
    return np.concatenate(con)

def run_packing():
    best_sum = -1.0
    best_solution = None
    bounds = [(0.0, 1.0)] * (N_CIRCLES * 2) + [(0.0, 0.5)]
    
    # Try multiple initial configurations to escape local minima
    for seed in range(5):
        rng = np.random.RandomState(seed + 42)
        
        # Initialize centers in a perturbed hexagonal pattern
        centers = np.zeros((N_CIRCLES, 2))
        idx = 0
        spacing = 0.18
        for row in range(8):
            y = 0.1 + row * spacing * np.sqrt(3)/2
            if y > 0.9: break
            for col in range(8):
                x = 0.1 + col * spacing + (row % 2) * spacing/2
                if x <= 0.9 and idx < N_CIRCLES:
                    centers[idx] = [x, y]
                    idx += 1
        
        # Add small random perturbation
        centers += rng.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        x0 = np.concatenate([centers.flatten(), [0.05]])
        
        try:
            res = minimize(
                objective_func, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': compute_constraints},
                options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                c_opt = res.x[:-1].reshape((N_CIRCLES, 2))
                r_opt = res.x[-1]
                
                # Apply a tiny safety margin to satisfy strict validation tolerances
                r_opt *= 0.99995
                radii = np.full(N_CIRCLES, r_opt)
                
                curr_sum = np.sum(radii)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_solution = (c_opt, radii, curr_sum)
                    
        except Exception:
            continue
            
    if best_solution is None:
        # Fallback to a simple grid if optimization fails
        c = np.zeros((N_CIRCLES, 2))
        for i in range(N_CIRCLES):
            c[i] = [0.1 + (i % 5) * 0.2, 0.1 + (i // 5) * 0.2]
        if N_CIRCLES > 25: 
            c[25] = [0.6, 0.9]
        return c, np.full(N_CIRCLES, 0.04), 1.04
        
    return best_solution
