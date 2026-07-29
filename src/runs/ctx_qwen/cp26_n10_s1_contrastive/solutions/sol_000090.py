# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000042 (state 26164787) state=e01611c4 sum of radii=2.625625 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def compute_obj(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2*N:])

def compute_constr(x):
    """
    Constraint function: ensures circles are inside [0,1]^2 and do not overlap.
    Returns a 1D array of values that must be >= 0.
    """
    c = []
    centers = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Boundary constraints
    c.append(centers[:, 0] - r)
    c.append(1.0 - centers[:, 0] - r)
    c.append(centers[:, 1] - r)
    c.append(1.0 - centers[:, 1] - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = centers[:, 0, np.newaxis] - centers[:, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    c.append(dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2)
    
    return np.concatenate(c)

def make_init_hex():
    """Generates a hexagonal lattice initialization."""
    rows = [6, 5, 6, 5, 4]
    pts = []
    r_est = 0.09
    y = r_est
    for k, cnt in enumerate(rows):
        x_off = 0.0 if k % 2 == 0 else r_est
        x = r_est + x_off
        for _ in range(cnt):
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
    pts = np.array(pts[:N])
    # Normalize to [0.1, 0.9] range
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    pts = (pts - mn) / (mx - mn) * 0.8 + 0.1
    return pts

def make_init_force(seed):
    """Generates a force-directed layout initialization."""
    np.random.seed(seed)
    pts = np.random.rand(N, 2)
    for _ in range(200):
        # Vectorized force computation
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff**2, axis=2)
        d2 = np.maximum(d2, 1e-6)
        d = np.sqrt(d2)
        F = diff / (d2[:, :, None] * d[:, :, None])
        forces = np.sum(F, axis=0)
        
        # Wall repulsion
        for k in range(2):
            wall_low = np.where(pts[:, k] < 0.05, 5.0 * (0.05 - pts[:, k]), 0.0)
            wall_high = np.where(pts[:, k] > 0.95, -5.0 * (pts[:, k] - 0.95), 0.0)
            forces[:, k] += wall_low + wall_high
            
        pts += 0.005 * forces
        pts = np.clip(pts, 0.01, 0.99)
    return pts

def get_safe_radii(centers):
    """Computes strictly feasible initial radii for given centers."""
    r = np.full(N, 0.0)
    for i in range(N):
        d_wall = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
        d_min = 2.0
        for j in range(N):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < d_min:
                    d_min = d
        r[i] = 0.4 * min(d_wall, 0.5 * d_min)
    return np.maximum(r, 1e-6)

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constr}
    
    best_x = None
    best_val = -np.inf
    
    inits = []
    
    # Hexagonal base with perturbations
    hex_pts = make_init_hex()
    inits.append(np.concatenate([hex_pts.flatten(), get_safe_radii(hex_pts)]))
    for seed in range(10):
        np.random.seed(seed)
        p = hex_pts + np.random.randn(N, 2) * 0.02
        p = np.clip(p, 0.02, 0.98)
        inits.append(np.concatenate([p.flatten(), get_safe_radii(p)]))
        
    # Force-directed layouts
    for seed in range(15):
        p = make_init_force(seed)
        inits.append(np.concatenate([p.flatten(), get_safe_radii(p)]))
        
    # Grid base with perturbations
    g_pts = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
    inits.append(np.concatenate([g_pts.flatten(), get_safe_radii(g_pts)]))
    for seed in range(5):
        np.random.seed(seed + 100)
        p = g_pts + np.random.randn(N, 2) * 0.02
        p = np.clip(p, 0.02, 0.98)
        inits.append(np.concatenate([p.flatten(), get_safe_radii(p)]))

    # Optimization loop
    for x0 in inits:
        try:
            res = minimize(compute_obj, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            c_val = compute_constr(res.x)
            if np.min(c_val) >= -1e-7:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # High-precision polishing
    if best_x is not None:
        try:
            res_p = minimize(compute_obj, best_x, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if np.min(compute_constr(res_p.x)) >= -1e-7:
                best_x = res_p.x
        except Exception:
            pass
            
    # Fallback (should not be reached given robust inits)
    if best_x is None:
        best_x = inits[0]
        
    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
