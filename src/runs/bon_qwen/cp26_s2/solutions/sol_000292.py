# sol_000292 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d1ce3e9) state=bbc5e5f7 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_constraints(x):
    n = N_CIRCLES
    c = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    
    cons = []
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons.append(dx*dx + dy*dy - (r[i] + r[j])**2)
            
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, same for y
    cons.extend(c[:, 0] - r)
    cons.extend(1.0 - c[:, 0] - r)
    cons.extend(c[:, 1] - r)
    cons.extend(1.0 - c[:, 1] - r)
    
    return np.array(cons)

def objective_func(x):
    # Maximize sum of radii -> minimize negative sum
    return -np.sum(x[2*N_CIRCLES:])

def run_packing():
    n = N_CIRCLES
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    cons = {'type': 'ineq', 'fun': compute_constraints}
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    np.random.seed(42)
    
    # Multiple restarts with perturbed grid initialization
    for trial in range(6):
        centers = []
        row = 0
        col = 0
        for i in range(n):
            # Staggered grid layout favors hexagonal packing density
            x = 0.15 + col * 0.18
            y = 0.15 + row * 0.22 + (col % 2) * 0.11
            # Add jitter to break symmetry and help escape local minima
            x += np.random.uniform(-0.03, 0.03)
            y += np.random.uniform(-0.03, 0.03)
            # Keep within reasonable bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
            
            col += 1
            if col > 4:
                col = 0
                row += 1
                
        centers = np.array(centers)
        radii = np.full(n, 0.085)
        x0 = np.concatenate([centers.flatten(), radii])
        
        try:
            res = minimize(
                objective_func, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 2500, 'ftol': 1e-11, 'disp': False}
            )
            
            curr_sum = np.sum(res.x[2*n:])
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x[:2*n].reshape(n, 2)
                best_radii = res.x[2*n:]
        except Exception:
            continue
            
    # Post-processing: ensure physical validity
    if best_radii is not None:
        best_radii = np.maximum(best_radii, 0.0)
        best_sum = float(np.sum(best_radii))
    else:
        # Fallback to safe grid packing
        grid_pts = np.linspace(0.12, 0.88, 5)
        best_centers = np.array([[x, y] for x in grid_pts for y in grid_pts][:n])
        best_radii = np.full(n, 0.07)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
