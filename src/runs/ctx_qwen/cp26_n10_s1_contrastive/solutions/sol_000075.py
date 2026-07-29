# sol_000075 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=747baf3c sum of radii=2.624565 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Compute all inequality constraints: boundary containment and pairwise separation."""
    C = x.reshape(N, 3)
    xc, yc, r = C[:, 0], C[:, 1], C[:, 2]
    
    c = [xc - r, 1.0 - xc - r, yc - r, 1.0 - yc - r]
    
    i, j = np.triu_indices(N, 1)
    dx = xc[i] - xc[j]
    dy = yc[i] - yc[j]
    rs = r[i] + r[j]
    c.append(dx*dx + dy*dy - rs*rs)
    
    return np.concatenate(c)

def make_hex_init(perturb_std):
    """Generates a hexagonal lattice initialization."""
    pts = []
    rows = [6, 5, 6, 5, 4]
    y = 0.1
    for r_idx, cnt in enumerate(rows):
        shift = 0.1 if r_idx % 2 == 1 else 0.0
        x = 0.1 + shift
        for _ in range(cnt):
            pts.append([x, y])
            x += 0.2
        y += 0.17320508075688772
    pts = np.array(pts[:N])
    if perturb_std > 0:
        pts += np.random.randn(N, 2) * perturb_std
    return np.clip(pts, 0.05, 0.95)

def make_grid_init(perturb_std):
    """Generates a perturbed grid initialization."""
    pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    pts = np.vstack([pts, [0.5, 0.5]])
    if perturb_std > 0:
        pts += np.random.randn(N, 2) * perturb_std
    return np.clip(pts, 0.05, 0.95)

def make_random_init(seed_val):
    """Generates a random initialization."""
    np.random.seed(seed_val)
    return np.random.rand(N, 2) * 0.8 + 0.1

def get_safe_radii(pts):
    """Computes strictly feasible initial radii for given centers."""
    n = len(pts)
    r = np.zeros(n)
    for i in range(n):
        d_bound = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        d_min = 1.0
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < d_min:
                    d_min = d
        r[i] = 0.35 * min(d_bound, 0.5 * d_min)
    return r

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_x = None
    best_sum = -np.inf
    
    # Phase 1: Diverse initializations
    inits = []
    for p in [0.0, 0.005, 0.01, 0.02, 0.04]:
        inits.append(make_hex_init(p))
    for p in [0.0, 0.005, 0.01, 0.02]:
        inits.append(make_grid_init(p))
    for s in range(15):
        inits.append(make_random_init(s))
        
    for pts in inits:
        r_safe = get_safe_radii(pts)
        x0 = np.zeros(3*N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_safe
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 4000, 'ftol': 1e-13})
            if res.success:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-8:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_x = res.x.copy()
        except Exception:
            continue

    # Phase 2: Scheduled perturbation local search
    if best_x is not None:
        for step in range(40):
            noise_std = 0.004 * (0.92**step)
            x_pert = best_x + np.random.randn(3*N) * noise_std
            x_pert[0::3] = np.clip(x_pert[0::3], 0.001, 0.999)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.001, 0.999)
            x_pert[2::3] = np.clip(x_pert[2::3], 0.001, 0.499)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons_dict, options={'maxiter': 3000, 'ftol': 1e-13})
                if res.success:
                    cv = constraints(res.x)
                    if np.min(cv) >= -1e-8:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_x = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-14})
            if res.success:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-9:
                    best_x = res.x
                    best_sum = -res.fun
        except Exception:
            pass

    # Fallback valid configuration
    if best_x is None:
        pts = make_hex_init(0.0)
        r_safe = get_safe_radii(pts)
        best_x = np.zeros(3*N)
        best_x[0::3] = pts[:, 0]
        best_x[1::3] = pts[:, 1]
        best_x[2::3] = r_safe
        best_sum = np.sum(r_safe)

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    radii = np.maximum(radii, 1e-9)
    
    # Final safety check to guarantee passing the 1e-12 validation tolerance
    cv = constraints(best_x)
    if np.min(cv) < -1e-11:
        scale = 1.0 - 1e-7
        best_x[2::3] *= scale
        radii = best_x[2::3]
        
    return centers, radii, float(np.sum(radii))
