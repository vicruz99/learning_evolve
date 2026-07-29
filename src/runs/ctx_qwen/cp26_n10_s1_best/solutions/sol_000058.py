# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000030 (state 57c93ce5) state=f3d60dee sum of radii=2.615535 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def unpack(vars):
    """Convert flat vector to x, y, r arrays."""
    return vars[0::3], vars[1::3], vars[2::3]

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Compute all inequality constraints >= 0 efficiently."""
    x, y, r = unpack(vars)
    n = N_CIRCLES
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    b = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    i, j = np.tril_indices(n, -1)
    o = dx[i, j]**2 + dy[i, j]**2 - dr[i, j]**2
    
    return np.concatenate([b, o])

def force_relax(centers, radii, steps=1500):
    """Force-directed layout to spread circles and push to boundaries."""
    n = len(centers)
    for _ in range(steps):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.hypot(dx, dy)
                if dist < 0.3 and dist > 1e-6:
                    f = 0.002 / (dist**2)
                    fx, fy = f * dx, f * dy
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
                    
        # Boundary repulsion
        for i in range(n):
            margin = radii[i] + 0.015
            if centers[i, 0] < margin: forces[i, 0] += 0.015
            if centers[i, 0] > 1.0 - margin: forces[i, 0] -= 0.015
            if centers[i, 1] < margin: forces[i, 1] += 0.015
            if centers[i, 1] > 1.0 - margin: forces[i, 1] -= 0.015
            
        centers += forces * 0.05
        centers = np.clip(centers, 0.005, 0.995)
        
        # Gradually expand radii to force denser packing
        radii = np.clip(radii * 1.0005, 0.01, 0.4)
        
    # Assign final radii based on local geometry (leaving slack for optimizer)
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d * 0.46
        
    return centers, radii

def get_init(n, seed, method='hex'):
    """Generate initial configuration and run force relaxation."""
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    if method == 'hex':
        r0 = 0.085
        idx = 0
        y = r0
        row = 0
        while idx < n:
            x = r0 if row % 2 == 0 else 2 * r0
            while idx < n and x + r0 <= 1.0:
                centers[idx] = [x + np.random.uniform(-0.01, 0.01), 
                                y + np.random.uniform(-0.01, 0.01)]
                idx += 1
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        while idx < n:
            centers[idx] = np.random.uniform(0.1, 0.9, 2)
            idx += 1
        radii[:] = r0
    elif method == 'grid':
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx < n:
                    centers[idx] = [0.1 + c * 0.18 + np.random.uniform(-0.01, 0.01), 
                                    0.1 + r * 0.16 + np.random.uniform(-0.01, 0.01)]
                    idx += 1
        while idx < n:
            centers[idx] = np.random.uniform(0.15, 0.85, 2)
            idx += 1
        radii[:] = 0.075
    else:  # random
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        radii[:] = 0.06
        
    centers, radii = force_relax(centers, radii, steps=1500)
    
    x0 = np.zeros(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Phase 1: Diverse initializations
    configs = [
        ('hex', 0), ('hex', 1), ('hex', 2), ('hex', 3),
        ('grid', 0), ('grid', 1), 
        ('rand', 0), ('rand', 1), ('rand', 2)
    ]
    
    for method, seed in configs:
        x0 = get_init(n, seed=seed, method=method)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            vals = constraints(res.x)
            if s > best_sum and np.min(vals) >= -1e-7:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement to escape shallow minima
    if best_vars is not None:
        for _ in range(6):
            x0 = best_vars + np.random.normal(0, 5e-5, 3 * n)
            for k in range(n):
                r = max(0.0, x0[3 * k + 2])
                x0[3 * k] = np.clip(x0[3 * k], r, 1.0 - r)
                x0[3 * k + 1] = np.clip(x0[3 * k + 1], r, 1.0 - r)
                x0[3 * k + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 3000, 'ftol': 1e-14})
                s = -res.fun
                if s > best_sum and np.min(constraints(res.x)) >= -1e-7:
                    best_sum = s
                    best_vars = res.x.copy()
            except Exception:
                break
                
    # Fallback
    if best_vars is None:
        best_vars = get_init(n, 0, 'hex')
        best_sum = -np.sum(best_vars[2::3])
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_vars[3 * i]
        centers[i, 1] = best_vars[3 * i + 1]
        radii[i] = best_vars[3 * i + 2]
        
    # Final validation & safety repair
    valid = True
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if r < 0 or x - r < -1e-10 or x + r > 1 + 1e-10 or y - r < -1e-10 or y + r > 1 + 1e-10:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        factor = 0.995
        for _ in range(50):
            radii *= factor
            centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-12:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
                
    return centers, radii, float(np.sum(radii))
