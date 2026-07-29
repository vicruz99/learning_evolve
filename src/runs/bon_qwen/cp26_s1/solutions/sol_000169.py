# sol_000169 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6d8d18a8) state=188c3896 sum of radii=2.610847 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import functools

N_CIRCLES = 26

def inter_constraint(vars, i, j):
    """Enforces non-overlap between circle i and j: dist^2 >= (ri + rj)^2"""
    xi, yi = vars[2*i], vars[2*i+1]
    xj, yj = vars[2*j], vars[2*j+1]
    ri, rj = vars[2*N_CIRCLES+i], vars[2*N_CIRCLES+j]
    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2

def bound_constraint(vars, i, bt):
    """Enforces boundary constraints: 0 <= center - r and center + r <= 1"""
    xi, yi = vars[2*i], vars[2*i+1]
    ri = vars[2*N_CIRCLES+i]
    if bt == 0: return xi - ri          # xmin
    if bt == 1: return 1.0 - (xi + ri)  # xmax
    if bt == 2: return yi - ri          # ymin
    if bt == 3: return 1.0 - (yi + ri)  # ymax

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Generate initial guess: Hexagonal lattice pattern
    # Rows configuration to total 26 circles
    row_counts = [5, 5, 5, 5, 6]
    centers_list = []
    
    y_pos = 0.05
    dy = 0.9 / 4.0  # 4 gaps for 5 rows
    
    for r_idx, count in enumerate(row_counts):
        if count == 6:
            dx = 0.9 / 5.0
        else:
            dx = 0.9 / 4.0
            
        x_start = 0.05 if r_idx % 2 == 0 else 0.05 + dx / 2.0
        
        for c in range(count):
            cx = x_start + c * dx
            cy = y_pos
            centers_list.append([cx, cy])
        y_pos += dy
        
    centers_init = np.array(centers_list[:N_CIRCLES])
    radii_init = np.full(N_CIRCLES, 0.05)
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # 2. Define constraints
    cons = []
    # Inter-circle constraints
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            cons.append({'type': 'ineq', 'fun': functools.partial(inter_constraint, i=i, j=j)})
            
    # Boundary constraints
    for i in range(N_CIRCLES):
        for bt in range(4):
            cons.append({'type': 'ineq', 'fun': functools.partial(bound_constraint, i=i, bt=bt)})
            
    # 3. Bounds for variables
    bounds = [(0.0, 1.0) for _ in range(2 * N_CIRCLES)] + [(1e-8, 0.5) for _ in range(N_CIRCLES)]
    
    # 4. Objective: maximize sum of radii -> minimize negative sum
    def objective(vars):
        return -np.sum(vars[2 * N_CIRCLES:])
        
    # 5. Optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
    )
    
    final_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    final_radii = res.x[2 * N_CIRCLES:]
    
    # Ensure strict validity against numerical noise
    # Clip radii to be strictly positive and within bounds
    final_radii = np.clip(final_radii, 1e-8, 0.5)
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1.0 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1.0 - final_radii)
    
    return final_centers, final_radii, float(np.sum(final_radii))
