# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 55285a70) state=37876f6a sum of radii=2.277828 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_constraints(v):
    con = []
    # Boundary constraints for each circle
    for i in range(N_CIRCLES):
        idx = i * 3
        con.append(v[idx] - v[idx+2])           # x - r >= 0
        con.append(1.0 - v[idx] - v[idx+2])     # 1 - x - r >= 0  => x + r <= 1
        con.append(v[idx+1] - v[idx+2])         # y - r >= 0
        con.append(1.0 - v[idx+1] - v[idx+2])   # 1 - y - r >= 0  => y + r <= 1
        
    # Pairwise non-overlap constraints
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            idx_i = i * 3
            idx_j = j * 3
            dx = v[idx_i] - v[idx_j]
            dy = v[idx_i+1] - v[idx_j+1]
            r_sum = v[idx_i+2] + v[idx_j+2]
            con.append(dx*dx + dy*dy - r_sum*r_sum)  # dist^2 >= (r1+r2)^2
            
    return np.array(con)

def objective_func(v):
    # We minimize negative sum of radii => maximizes sum of radii
    return -np.sum(v[2::3])

def get_initial_config():
    centers = []
    radii = []
    # Hexagonal lattice spacing
    dx = 0.16
    dy = 0.16 * np.sqrt(3) / 2.0
    row = 0
    count = 0
    while count < N_CIRCLES:
        y_val = 0.1 + row * dy
        x_start = 0.1 + (row % 2) * dx / 2.0
        for c in range(8):
            if count >= N_CIRCLES:
                break
            x_val = x_start + c * dx
            if x_val <= 0.9 and y_val <= 0.9:
                centers.append([x_val, y_val])
                radii.append(0.05)
                count += 1
        row += 1
    return np.array(centers), np.array(radii)

def run_packing():
    centers, radii = get_initial_config()
    x0 = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Small perturbation to break symmetry and aid convergence
    np.random.seed(42)
    x0 += np.random.normal(0, 0.002, size=x0.shape)
    
    # Ensure initial feasibility w.r.t bounds
    x0[:2*N_CIRCLES:3] = np.clip(x0[:2*N_CIRCLES:3], 0.02, 0.98)
    x0[1:2*N_CIRCLES:3] = np.clip(x0[1:2*N_CIRCLES:3], 0.02, 0.98)
    x0[2::3] = np.clip(x0[2::3], 0.02, 0.4)
    
    bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    result = minimize(objective_func, x0, method='SLSQP', bounds=bounds, 
                      constraints=cons, options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
    
    final_v = result.x
    final_centers = np.zeros((N_CIRCLES, 2))
    final_radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        final_centers[i, 0] = final_v[3*i]
        final_centers[i, 1] = final_v[3*i+1]
        final_radii[i] = final_v[3*i+2]
        
    # Safety clamp for numerical precision
    final_radii = np.maximum(final_radii, 0.0)
    
    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii
