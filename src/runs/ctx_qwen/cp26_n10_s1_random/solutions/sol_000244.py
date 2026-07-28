# sol_000244 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000233 (state 4b6f20f2) state=254c66d0 sum of radii=2.553812 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def constraints_equal(vars_arr, n):
    """Inequality constraints for equal-radius packing: must be >= 0."""
    x = vars_arr[:n]
    y = vars_arr[n:2*n]
    t = vars_arr[-1]
    c = []
    c.append(x - t)
    c.append(1.0 - x - t)
    c.append(y - t)
    c.append(1.0 - y - t)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    c.append(dx**2 + dy**2 - 4.0 * t**2)
    return np.concatenate(c)

def obj_equal(vars_arr):
    """Objective: maximize equal radius t."""
    return -vars_arr[-1]

def constraints_unequal(vars_arr, n):
    """Inequality constraints for unequal-radius packing: must be >= 0."""
    x = vars_arr[:n]
    y = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    c = []
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dr = r[idx_i] + r[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c)

def obj_unequal(vars_arr):
    """Objective: maximize sum of radii."""
    return -np.sum(vars_arr[2*26:])

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A = np.zeros((m, n))
    A[np.arange(m), idx_i] = 1.0
    A[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b = dists[idx_i, idx_j]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def hex_init(pattern, n=26):
    """Generates initial hexagonal lattice configuration."""
    pts = []
    r0 = 0.10
    y = r0
    for ri, cnt in enumerate(pattern):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3)
    pts = np.array(pts[:n])
    # Normalize to fit comfortably inside [0,1]
    c_min = pts.min(axis=0)
    c_max = pts.max(axis=0)
    cfg = (pts - c_min) / (c_max - c_min) * 0.8 + 0.1
    return cfg

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], 
        [5,5,6,5,5], [4,6,6,6,4], [6,4,6,5,5],
        [5,6,4,6,5], [7,6,6,7]
    ]
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Equal radius optimization from structured starts
    for pat in patterns:
        if sum(pat) < n:
            continue
        cfg = hex_init(pat, n)
        
        x0 = np.concatenate([cfg.flatten(), [0.095]])
        bounds_eq = [(0.0, 1.0)]*(2*n) + [(0.05, 0.15)]
        
        try:
            res = minimize(obj_equal, x0, args=(), method='SLSQP', bounds=bounds_eq,
                           constraints={'type': 'ineq', 'fun': constraints_equal, 'args': (n,)},
                           options={'maxiter': 5000, 'ftol': 1e-12})
            # Update if we found a better equal radius configuration
            if -res.fun > best_sum / n:
                c_eq = res.x[:2*n].reshape(n, 2)
                r_lp, s_lp = solve_lp(c_eq)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_eq.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # Phase 2: Joint unequal optimization to refine positions and sizes
    if best_centers is not None:
        for _ in range(5):
            x0 = np.zeros(3*n)
            x0[:n] = best_centers[:,0]
            x0[n:2*n] = best_centers[:,1]
            x0[2*n:] = best_radii * 0.98
            
            x0[:2*n] += rng.uniform(-0.005, 0.005, 2*n)
            x0[:2*n] = np.clip(x0[:2*n], 0.01, 0.99)
            
            bounds_uneq = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
            try:
                res = minimize(obj_unequal, x0, method='SLSQP', bounds=bounds_uneq,
                               constraints={'type': 'ineq', 'fun': constraints_unequal, 'args': (n,)},
                               options={'maxiter': 8000, 'ftol': 1e-13})
                if np.isfinite(res.fun):
                    c_new = np.column_stack((res.x[:n], res.x[n:2*n]))
                    r_new, s_new = solve_lp(c_new)
                    if s_new > best_sum:
                        best_sum = s_new
                        best_centers = c_new.copy()
                        best_radii = r_new.copy()
            except Exception:
                pass

    # Phase 3: Hill climbing on centers with LP radius evaluation
    if best_centers is not None:
        for step in range(2000):
            i = rng.integers(n)
            old_c = best_centers[i].copy()
            decay = 0.94 ** (step / 40.0)
            shift = rng.uniform(-0.008, 0.008, 2) * decay
            best_centers[i] = np.clip(old_c + shift, 1e-4, 1.0 - 1e-4)
            
            r_try, s_try = solve_lp(best_centers)
            if s_try > best_sum:
                best_sum = s_try
                best_radii = r_try.copy()
            else:
                best_centers[i] = old_c
                
    # Fallback in case all optimizations fail
    if best_centers is None:
        best_centers = np.array([[0.5, 0.5]] * n)
        best_radii = np.full(n, 0.01)
        best_sum = 0.26
        
    # Final strict safety scaling to guarantee numerical validity within 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
