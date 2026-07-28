# sol_000190 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000163 (state 5ceb6a50) state=3119f088 sum of radii=2.514572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers, n):
    """Solves LP to maximize sum of radii for fixed centers."""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    b_ub = dists[idx_i, idx_j]
    
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def objective_slsqp(vars_arr, n):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_arr[2 * n:])

def constraints_slsqp(vars_arr, n):
    """Inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    
    # Boundary constraints
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    # Pairwise non-overlap constraints (squared for smooth gradients)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    idx = np.triu_indices(n, k=1)
    c = np.concatenate([c, (dx[idx]**2 + dy[idx]**2) - dr[idx]**2])
    return c

def generate_inits(n, rng):
    """Generates diverse high-quality initial center configurations."""
    inits = []
    
    # Hexagonal patterns with various row distributions summing to 26
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [5, 6, 6, 5, 4], [6, 6, 5, 5, 4],
        [5, 5, 5, 5, 6], [5, 7, 5, 5, 4]
    ]
    
    for pat in row_patterns:
        pts = []
        y = 0.06
        for r_idx, cnt in enumerate(pat):
            shift = 0.06 if r_idx % 2 == 1 else 0.0
            width = (cnt - 1) * 0.12
            x_start = 0.5 - width / 2.0 + shift
            for c in range(cnt):
                if len(pts) < n:
                    pts.append([x_start + c * 0.12, y])
            y += 0.104
            if len(pts) >= n: break
        inits.append(np.array(pts[:n]))
        
    # Random dense packs to escape structural biases
    for _ in range(10):
        pts = rng.uniform(0.1, 0.9, (n, 2))
        inits.append(pts)
        
    # Grid based initialization
    g = np.linspace(0.15, 0.85, 5)
    grid = np.array([[x, y] for y in g for x in g])
    grid = np.vstack([grid, [0.5, 0.5]])
    inits.append(grid)
    
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints_slsqp, 'args': (n,)}
    
    rng = np.random.default_rng(42)
    inits = generate_inits(n, rng)
    
    # Phase 1: Multi-start SLSQP optimization
    for cfg in inits:
        v0 = np.zeros(3 * n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = 0.02  # Start with small feasible radii
        
        try:
            res = minimize(objective_slsqp, v0, args=(n,), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            
            if np.isfinite(res.fun):
                cx, cy, r = res.x[:n], res.x[n:2*n], res.x[2*n:]
                valid = True
                if np.any(cx < r - 1e-8) or np.any(cx > 1.0 - r + 1e-8) or \
                   np.any(cy < r - 1e-8) or np.any(cy > 1.0 - r + 1e-8):
                    valid = False
                if valid:
                    d2 = (cx[:, None] - cx[None, :])**2 + (cy[:, None] - cy[None, :])**2
                    rs = (r[:, None] + r[None, :])**2
                    idx = np.triu_indices(n, k=1)
                    if np.any(d2[idx] < rs[idx] - 1e-8):
                        valid = False
                
                if valid:
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = np.column_stack((cx, cy))
                        best_radii = r.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative LP-SLSQP Refinement to escape local minima
    if best_centers is not None:
        for _ in range(6):
            radii_lp, s_lp = solve_lp_radii(best_centers, n)
            if radii_lp is not None:
                v0_ref = np.zeros(3 * n)
                v0_ref[:n] = best_centers[:, 0]
                v0_ref[n:2*n] = best_centers[:, 1]
                v0_ref[2*n:] = np.maximum(radii_lp * 0.95, 1e-5)
                
                try:
                    res_ref = minimize(objective_slsqp, v0_ref, args=(n,), method='SLSQP',
                                       bounds=bounds, constraints=cons,
                                       options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                    if np.isfinite(res_ref.fun):
                        cx, cy, r = res_ref.x[:n], res_ref.x[n:2*n], res_ref.x[2*n:]
                        valid = True
                        if np.any(cx < r - 1e-8) or np.any(cx > 1.0 - r + 1e-8) or \
                           np.any(cy < r - 1e-8) or np.any(cy > 1.0 - r + 1e-8):
                            valid = False
                        if valid:
                            d2 = (cx[:, None] - cx[None, :])**2 + (cy[:, None] - cy[None, :])**2
                            rs = (r[:, None] + r[None, :])**2
                            idx = np.triu_indices(n, k=1)
                            if np.any(d2[idx] < rs[idx] - 1e-8):
                                valid = False
                            
                        if valid:
                            s = np.sum(r)
                            if s > best_sum:
                                best_sum = s
                                best_centers = np.column_stack((cx, cy))
                                best_radii = r.copy()
                except Exception:
                    pass
                    
            # Perturb best centers slightly for next refinement iteration
            best_centers += rng.uniform(-0.004, 0.004, best_centers.shape)
            best_centers = np.clip(best_centers, 0.05, 0.95)

    # Fallback configuration if optimization fails unexpectedly
    if best_centers is None:
        best_centers = inits[0]
        best_radii, _ = solve_lp_radii(best_centers, n)
        best_radii = best_radii if best_radii is not None else np.full(n, 0.08)
        best_sum = np.sum(best_radii)

    # Final safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
