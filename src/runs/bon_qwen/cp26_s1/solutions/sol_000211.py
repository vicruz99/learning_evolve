# sol_000211 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dddb8969) state=737bbdf8 sum of radii=2.622944 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    # v contains [x0, y0, r0, x1, y1, r1, ...]
    # We want to maximize sum(r), so minimize -sum(r)
    return -np.sum(v[2::3])

def constraints(v):
    # Returns array of constraint values. All must be >= 0.
    c = []
    # Boundary constraints: circles must stay inside [0,1]^2
    for i in range(N_CIRCLES):
        idx = 3 * i
        x, y, r = v[idx], v[idx+1], v[idx+2]
        c.extend([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
        
    # Non-overlap constraints: distance between centers >= sum of radii
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            idx_i, idx_j = 3 * i, 3 * j
            dx = v[idx_i] - v[idx_j]
            dy = v[idx_i+1] - v[idx_j+1]
            dr = v[idx_i+2] + v[idx_j+2]
            c.append(dx*dx + dy*dy - dr*dr)
            
    return np.array(c)

def get_initial_guess():
    # Generate a structured grid-like initial guess that is feasible
    guess = []
    cols = 5
    rows = 6
    step_x = 0.18
    step_y = 0.18
    x_start = 0.11
    y_start = 0.11
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N_CIRCLES:
                break
            x = x_start + c * step_x
            y = y_start + r * step_y
            guess.extend([x, y, 0.08])
            count += 1
        if count >= N_CIRCLES:
            break
    return np.array(guess)

def run_packing():
    # Bounds for x, y, r
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -np.inf
    
    # Multi-start optimization to escape local minima
    for seed in range(15):
        np.random.seed(seed)
        v0 = get_initial_guess()
        
        # Add controlled random perturbation
        noise = np.random.normal(0, 0.015, v0.shape)
        v0 = v0 + noise
        
        # Project to valid bounds
        v0 = np.clip(v0, 0.0, 1.0)
        v0[2::3] = np.clip(v0[2::3], 0.01, 0.5)
        
        res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
        
        curr_sum = -res.fun
        if curr_sum > best_sum:
            # Check if constraints are satisfied within numerical tolerance
            c_vals = constraints(res.x)
            if np.min(c_vals) > -1e-5:
                best_v = res.x.copy()
                best_sum = curr_sum
                
    # Assemble output arrays
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3]
    return centers, radii, float(best_sum)
