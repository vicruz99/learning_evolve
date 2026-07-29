# sol_000310 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=d8b4d562 sum of radii=2.559496 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[2*N:])

def constraint_overlap(vars):
    """Non-overlap constraints: dist^2 >= (r_i + r_j)^2"""
    centers = vars[:2*N].reshape(N, 2)
    r = vars[2*N:]
    cons = np.empty(N*(N-1)//2)
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            d2 = np.sum((centers[i] - centers[j])**2)
            cons[k] = d2 - (r[i] + r[j])**2
            k += 1
    return cons

def constraint_boundary(vars):
    """Boundary constraints: circles must stay inside [0,1]^2"""
    centers = vars[:2*N].reshape(N, 2)
    r = vars[2*N:]
    cons = np.empty(4*N)
    for i in range(N):
        x, y = centers[i]
        ri = r[i]
        idx = 4*i
        cons[idx] = x - ri
        cons[idx+1] = 1.0 - x - ri
        cons[idx+2] = y - ri
        cons[idx+3] = 1.0 - y - ri
    return cons

def run_packing():
    # 1. Hexagonal-like initial placement
    centers = np.zeros((N, 2))
    idx = 0
    y_coords = np.linspace(0.12, 0.88, 6)
    
    for row in range(6):
        y = y_coords[row]
        # Alternate 5 and 4 columns to approximate hexagonal packing
        n_cols = 5 if row % 2 == 0 else 4
        x_coords = np.linspace(0.1, 0.9, n_cols)
        if row % 2 != 0:
            x_coords += 0.1  # Shift odd rows to fill gaps
        for x in x_coords:
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
                
    # 2. Add small random perturbation to break symmetry and avoid flat landscapes
    np.random.seed(42)
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # 3. Setup optimization variables and bounds
    radii = np.full(N, 0.08)  # Initial feasible radius
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0, 1)]*(2*N) + [(0, None)]*N  # x,y in [0,1], r >= 0
    
    cons = [
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_boundary}
    ]
    
    # 4. Run SLSQP optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10})
                   
    final_centers = res.x[:2*N].reshape(N, 2)
    final_radii = res.x[2*N:]
    final_radii = np.maximum(final_radii, 1e-9)  # Ensure non-negative
    
    # 5. Project centers to strictly satisfy boundary constraints (handling numerical tolerance)
    for i in range(N):
        r = final_radii[i]
        final_centers[i, 0] = np.clip(final_centers[i, 0], r, 1.0 - r)
        final_centers[i, 1] = np.clip(final_centers[i, 1], r, 1.0 - r)
        
    return final_centers, final_radii, np.sum(final_radii)
