# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000036 (state 025191a3) state=cbb0cfcf sum of radii=1.950000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Compute all inequality constraints g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    cons = []
    # Boundary constraints
    cons.append(cx - cr)
    cons.append(1.0 - cx - cr)
    cons.append(cy - cr)
    cons.append(1.0 - cy - cr)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dr = cr[i] + cr[j]
            cons.append(dx*dx + dy*dy - dr*dr)
            
    return np.concatenate(cons)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return bounds

def make_init(seed, method='hex'):
    """Generate a feasible initial configuration."""
    np.random.seed(seed)
    cx = np.zeros(N_CIRCLES)
    cy = np.zeros(N_CIRCLES)
    cr = np.full(N_CIRCLES, 0.075)
    
    if method == 'hex':
        r = 0.085
        y = r
        row = 0
        idx = 0
        while idx < N_CIRCLES:
            x = r if row % 2 == 0 else 2 * r
            while idx < N_CIRCLES and x + r <= 1.0:
                cx[idx] = x
                cy[idx] = y
                idx += 1
                x += 2 * r
            y += np.sqrt(3) * r
            row += 1
            if y + r > 1.0:
                break
    else:
        cx = np.random.uniform(0.15, 0.85, N_CIRCLES)
        cy = np.random.uniform(0.15, 0.85, N_CIRCLES)
        
    # Perturb to break exact symmetry
    cx += np.random.normal(0, 0.004, N_CIRCLES)
    cy += np.random.normal(0, 0.004, N_CIRCLES)
    
    # Ensure strict feasibility for SLSQP
    cr = np.clip(cr, 0.05, 0.2)
    cx = np.clip(cx, cr, 1.0 - cr)
    cy = np.clip(cy, cr, 1.0 - cr)
    
    x0 = np.zeros(3 * N_CIRCLES)
    x0[0::3] = cx
    x0[1::3] = cy
    x0[2::3] = cr
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_val = -np.inf
    best_x = None
    
    # Phase 1: Multi-start optimization
    for seed in range(30):
        for method in ['hex', 'rand']:
            x0 = make_init(seed, method)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints=cons, options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
                
                curr_val = -res.fun
                if curr_val > best_val:
                    # Check constraint satisfaction
                    c_vals = constraint_func(res.x)
                    if np.min(c_vals) >= -1e-9:
                        best_val = curr_val
                        best_x = res.x.copy()
            except Exception:
                continue
                
    # Phase 2: Local refinement to escape local minima
    if best_x is not None:
        for k in range(20):
            # Perturb centers and radii
            x_pert = best_x + np.random.normal(0, 1.5e-4, 3 * N_CIRCLES)
            
            # Project to feasible bounds
            for i in range(N_CIRCLES):
                r = max(1e-6, x_pert[3*i+2])
                x_pert[3*i] = np.clip(x_pert[3*i], r, 1.0 - r)
                x_pert[3*i+1] = np.clip(x_pert[3*i+1], r, 1.0 - r)
                
            # Occasionally try expanding radii slightly to push boundaries
            if k % 3 == 0:
                x_pert[2::3] *= 1.002
                x_pert[0::3] = np.clip(x_pert[0::3], x_pert[2::3], 1.0 - x_pert[2::3])
                x_pert[1::3] = np.clip(x_pert[1::3], x_pert[2::3], 1.0 - x_pert[2::3])
                
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
                
                curr_val = -res.fun
                if curr_val > best_val:
                    c_vals = constraint_func(res.x)
                    if np.min(c_vals) >= -1e-9:
                        best_val = curr_val
                        best_x = res.x.copy()
            except Exception:
                continue

    # Fallback initialization if optimization failed entirely
    if best_x is None:
        best_x = make_init(0, 'hex')
        
    # Extract results
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    centers[:, 0] = best_x[0::3]
    centers[:, 1] = best_x[1::3]
    radii[:] = best_x[2::3]
    
    # Final safety check and minimal shrinkage to guarantee strict validity
    valid = False
    for _ in range(100):
        valid = True
        for i in range(N_CIRCLES):
            if radii[i] < 0:
                valid = False; break
            if centers[i,0] - radii[i] < -1e-12 or centers[i,0] + radii[i] > 1.0 + 1e-12:
                valid = False; break
            if centers[i,1] - radii[i] < -1e-12 or centers[i,1] + radii[i] > 1.0 + 1e-12:
                valid = False; break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                    if d < radii[i] + radii[j] - 1e-12:
                        valid = False; break
                if not valid:
                    break
        if valid:
            break
        radii *= 0.9995
        # Re-clip centers if radii shrank
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
