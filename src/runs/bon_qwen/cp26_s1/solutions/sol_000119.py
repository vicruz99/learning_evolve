# sol_000119 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 47219f56) state=8514c8b2 sum of radii=2.602920 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective_func(v):
    # Maximize sum of radii -> minimize negative sum
    return -np.sum(v[2::3])

def constraint_func(v):
    # Returns array of constraint values. Must be >= 0 for validity.
    res = []
    for i in range(N):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        
        # Boundary constraints: circle inside [0,1]x[0,1]
        res.append(x - r)
        res.append(1.0 - x - r)
        res.append(y - r)
        res.append(1.0 - y - r)
        
        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        for j in range(i + 1, N):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dr = v[3*i+2] + v[3*j+2]
            res.append(dx*dx + dy*dy - dr*dr)
            
    return np.array(res)

def get_initial_config():
    # Hexagonal lattice initialization
    centers = []
    radii = []
    r_init = 0.08
    dy = np.sqrt(3)/2 * 2 * r_init
    row_counts = [5, 6, 5, 6, 4]
    
    for r_idx, cnt in enumerate(row_counts):
        y = 0.5 - (len(row_counts) - 1) * dy / 2 + r_idx * dy
        shift = r_init if r_idx % 2 == 1 else 0.0
        row_width = cnt * 2 * r_init
        x_start = 0.5 - row_width / 2 + shift
        
        for c in range(cnt):
            centers.append([x_start + c * 2 * r_init, y])
            radii.append(r_init)
            
    return np.array(centers), np.array(radii)

def run_packing():
    centers, radii = get_initial_config()
    v0 = np.zeros(3 * N)
    
    for i in range(N):
        v0[3*i] = centers[i, 0]
        v0[3*i+1] = centers[i, 1]
        v0[3*i+2] = radii[i]
        
    # Small perturbation to break symmetry and avoid flat valleys
    np.random.seed(42)
    v0 += np.random.uniform(-1e-5, 1e-5, size=v0.shape)
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Stage 1: Coarse optimization
    res1 = minimize(objective_func, v0, method='SLSQP', bounds=bounds, 
                    constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10})
                    
    # Stage 2: Fine-tuning for strict feasibility
    res2 = minimize(objective_func, res1.x, method='SLSQP', bounds=bounds, 
                    constraints=cons, options={'maxiter': 2000, 'ftol': 1e-14})
                    
    v_opt = res2.x
    
    # Extract results
    opt_centers = np.array([[v_opt[3*i], v_opt[3*i+1]] for i in range(N)])
    opt_radii = v_opt[2::3]
    total_sum = np.sum(opt_radii)
    
    return opt_centers, opt_radii, total_sum
