# sol_000117 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 028484b6) state=3370278b sum of radii=2.621357 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars):
    """Objective: minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(vars[52:])

def constraints_func(vars):
    """Inequality constraints: boundaries and non-overlap."""
    N = 26
    centers = vars[:2*N].reshape((N, 2))
    radii = vars[2*N:]
    cons = []
    
    # Boundary and non-negativity constraints
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        cons.extend([x - r, 1.0 - x - r, y - r, 1.0 - y - r, r])
        
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            rs = radii[i] + radii[j]
            cons.append(dx * dx + dy * dy - rs * rs)
            
    return np.array(cons)

def run_packing():
    N = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Variable bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Multiple restarts to avoid local minima
    num_trials = 20
    for trial in range(num_trials):
        rng = np.random.RandomState(1000 + trial)
        
        # Initialize centers on a perturbed grid to ensure strict feasibility
        centers_init = np.zeros((N, 2))
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx >= N:
                    break
                centers_init[idx, 0] = 0.15 + c * 0.16
                centers_init[idx, 1] = 0.15 + r * 0.16
                idx += 1
                
        # Add random perturbation to break symmetry
        centers_init += rng.uniform(-0.03, 0.03, centers_init.shape)
        centers_init = np.clip(centers_init, 0.05, 0.95)
        
        # Initialize radii small enough to be strictly feasible with the grid
        radii_init = 0.04 + rng.rand(N) * 0.01
        
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints_func},
                           options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                val = -res.fun
                if val > best_sum:
                    best_sum = val
                    best_centers = res.x[:2*N].reshape((N, 2))
                    best_radii = res.x[2*N:]
        except Exception:
            continue
            
    # Fallback in case optimization fails completely
    if best_centers is None:
        centers_fb = np.zeros((N, 2))
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx >= N: 
                    break
                centers_fb[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                idx += 1
        radii_fb = np.ones(N) * 0.08
        return centers_fb, radii_fb, np.sum(radii_fb)
        
    # Post-processing to guarantee strict validity against numerical errors
    min_gap = 1.0
    for i in range(N):
        x, y = best_centers[i]
        r = best_radii[i]
        d = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        if d < min_gap:
            min_gap = d
            
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_centers[i, 0] - best_centers[j, 0]
            dy = best_centers[i, 1] - best_centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            d = dist - (best_radii[i] + best_radii[j])
            if d < min_gap:
                min_gap = d
                
    # Shrink uniformly if any constraint is marginally violated
    if min_gap < 0.0:
        shrink = -min_gap + 1e-10
        best_radii = np.maximum(0.0, best_radii - shrink)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
