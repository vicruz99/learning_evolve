# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000021 (state 2060a481) state=e8ca52ab sum of radii=2.595357 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    # Minimize negative sum of radii
    # Radii are at indices 2, 5, 8, ..., i.e., vars[2::3]
    return -np.sum(vars[2::3])

def compute_constraints(vars):
    # Returns a vector of inequality constraints g(x) >= 0
    cons = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(N):
        xi = vars[3*i]
        yi = vars[3*i+1]
        ri = vars[3*i+2]
        cons.append(xi - ri)
        cons.append(1.0 - xi - ri)
        cons.append(yi - ri)
        cons.append(1.0 - yi - ri)
        
    # Overlap constraints: dist(i, j) - ri - rj >= 0
    for i in range(N):
        xi = vars[3*i]
        yi = vars[3*i+1]
        ri = vars[3*i+2]
        for j in range(i + 1, N):
            xj = vars[3*j]
            yj = vars[3*j+1]
            rj = vars[3*j+2]
            dx = xi - xj
            dy = yi - yj
            cons.append(np.sqrt(dx*dx + dy*dy) - ri - rj)
            
    return np.array(cons)

def generate_initial_guess(seed, method='hex'):
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    if method == 'hex':
        r_init = 0.085
        radii[:] = r_init
        pts = []
        k = 0
        while True:
            y = r_init + k * np.sqrt(3) * r_init
            if y > 1.0 + r_init: break
            m = 0
            while True:
                shift = r_init if k % 2 == 1 else 0
                x = r_init + shift + m * 2 * r_init
                if x > 1.0 + r_init: break
                pts.append([x, y])
                m += 1
            k += 1
            
        valid_pts = [p for p in pts if r_init <= p[0] <= 1.0 - r_init and r_init <= p[1] <= 1.0 - r_init]
        if len(valid_pts) < N:
            for _ in range(N - len(valid_pts)):
                valid_pts.append(np.random.uniform(r_init, 1.0 - r_init, 2))
                
        valid_pts = np.array(valid_pts)
        dists = np.linalg.norm(valid_pts - 0.5, axis=1)
        idx = np.argsort(dists)[:N]
        centers = valid_pts[idx] + np.random.normal(0, 0.005, (N, 2))
        
    elif method == 'grid':
        r_init = 0.075
        radii[:] = r_init
        idx = 0
        step = 0.16
        for r in range(7):
            for c in range(7):
                if idx < N:
                    cx = r_init + c * step
                    cy = r_init + r * step
                    if cx <= 1.0 - r_init and cy <= 1.0 - r_init:
                        centers[idx] = [cx, cy]
                        idx += 1
        while idx < N:
            centers[idx] = np.random.uniform(r_init, 1.0 - r_init, 2)
            idx += 1
        centers += np.random.normal(0, 0.005, (N, 2))
        
    else: # random
        r_init = 0.06
        radii[:] = r_init
        for i in range(N):
            centers[i] = np.random.uniform(r_init, 1.0 - r_init, 2)
        centers += np.random.normal(0, 0.005, (N, 2))
        
    centers = np.clip(centers, 0.01, 0.99)
    
    vars = np.zeros(3 * N)
    vars[0::3] = centers[:, 0]
    vars[1::3] = centers[:, 1]
    vars[2::3] = radii
    return vars

def run_packing():
    best_sum = -1.0
    best_vars = None
    
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    constraints = {'type': 'ineq', 'fun': compute_constraints}
    
    # Try multiple initial configurations to escape local minima
    methods = ['hex'] * 6 + ['grid'] * 4 + ['random'] * 10
    
    for idx, method in enumerate(methods):
        seed = idx * 13 + 7
        x0 = generate_initial_guess(seed, method)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun) and -res.fun > best_sum:
                best_sum = -res.fun
                best_vars = res.x.copy()
        except Exception:
            continue
            
    # Refinement: perturb best solution and optimize again to fine-tune
    if best_vars is not None:
        for _ in range(15):
            x0 = best_vars + np.random.normal(0, 0.0015, 3*N)
            for i in range(N):
                x0[3*i] = np.clip(x0[3*i], 0.0, 1.0)
                x0[3*i+1] = np.clip(x0[3*i+1], 0.0, 1.0)
                x0[3*i+2] = np.clip(x0[3*i+2], 0.0, 0.5)
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=constraints,
                               options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun) and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_vars = res.x.copy()
            except Exception:
                pass

    # Extract results
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(N):
        centers[i, 0] = best_vars[3*i]
        centers[i, 1] = best_vars[3*i+1]
        radii[i] = best_vars[3*i+2]
        
    # Safety shrinkage to guarantee validity against float errors
    valid = True
    for i in range(N):
        x, y, r = centers[i,0], centers[i,1], radii[i]
        if x < r or x > 1-r or y < r or y > 1-r:
            valid = False; break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if dist < radii[i] + radii[j] - 1e-11:
                    valid = False; break
            if not valid: break
            
    if not valid:
        factor = 1.0
        for _ in range(100):
            factor *= 0.995
            test_radii = radii * factor
            ok = True
            for i in range(N):
                x, y, r = centers[i,0], centers[i,1], test_radii[i]
                if x < r or x > 1-r or y < r or y > 1-r:
                    ok = False; break
            if ok:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                        if dist < test_radii[i] + test_radii[j] - 1e-11:
                            ok = False; break
                    if not ok: break
            if ok:
                radii = test_radii
                break

    return centers, radii, float(np.sum(radii))
