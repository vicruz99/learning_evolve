# sol_000277 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000262 (state 4217c70f) state=4f5e1e11 sum of radii=2.319999 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers and returns gradient."""
    n = centers.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    
    # Constraint matrix: r_i + r_j <= dist_ij
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    # Boundary limits: r_i <= min(x, 1-x, y, 1-y)
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        return None, None, None
        
    r = res.x
    s = -res.fun
    
    # Compute gradient using LP dual variables (marginals)
    grad = np.zeros((n, 2))
    try:
        lams = np.asarray(res.ineqlin.marginals)
        valid = lams > 1e-8
        if np.any(valid):
            mask = np.where(valid)[0]
            i_a = idx_i[mask]
            j_a = idx_j[mask]
            d_a = b_ub[mask]
            lam = lams[mask]
            d_safe = np.maximum(d_a, 1e-12)
            diff_vec = centers[i_a] - centers[j_a]
            factors = lam[:, np.newaxis] / d_safe[:, np.newaxis]
            contrib = diff_vec * factors
            np.add.at(grad, i_a, contrib)
            np.add.at(grad, j_a, -contrib)
    except Exception:
        pass
    return r, s, grad

def lp_obj_grad(x_flat):
    """Objective and gradient for L-BFGS-B: maximize LP sum of radii."""
    c = x_flat.reshape(N, 2)
    r, s, g = solve_lp(c)
    if r is None:
        return 1e6, np.full_like(x_flat, 0.0)
    return -s, -g.flatten()

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    starts = []
    
    # 1. Hexagonal lattice patterns
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,5,5],
        [6,6,5,5,4], [7,5,5,5,4], [5,6,6,5,4], [6,5,5,6,4]
    ]
    for pat in patterns:
        if sum(pat) != N: continue
        pts = []
        r0 = 0.10
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        starts.append(np.array(pts[:N]))
        
    # 2. Random uniform starts
    for _ in range(20):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # 3. Corner-focused starts (often yield higher density)
    for _ in range(10):
        pts = []
        pts.append([0.12, 0.12])
        pts.append([0.88, 0.12])
        pts.append([0.12, 0.88])
        pts.append([0.88, 0.88])
        pts.extend(rng.uniform(0.25, 0.75, (N-4, 2)).tolist())
        starts.append(np.array(pts[:N]))

    # Optimization loop
    for cfg in starts:
        x0 = np.clip(cfg.flatten(), 0.02, 0.98)
        
        # Primary local optimization
        res = minimize(lp_obj_grad, x0, method='L-BFGS-B', jac=True,
                       bounds=[(0.01, 0.99)]*(2*N),
                       options={'maxiter': 5000, 'ftol': 1e-14})
                       
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt, _ = solve_lp(c_opt)
        
        if r_opt is not None and s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
        # Adaptive perturbation search to escape local minima
        for scale_pert in [0.02, 0.01, 0.005, 0.001]:
            for _ in range(15):
                x_pert = c_opt.flatten() + rng.uniform(-scale_pert, scale_pert, 2*N)
                x_pert = np.clip(x_pert, 0.02, 0.98)
                res2 = minimize(lp_obj_grad, x_pert, method='L-BFGS-B', jac=True,
                                bounds=[(0.01, 0.99)]*(2*N),
                                options={'maxiter': 2000, 'ftol': 1e-13})
                c_pert = res2.x.reshape(N, 2)
                r_pert, s_pert, _ = solve_lp(c_pert)
                if r_pert is not None and s_pert > best_sum:
                    best_sum = s_pert
                    best_centers = c_pert.copy()
                    best_radii = r_pert.copy()

    # Fallback
    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, (N, 2))
        best_radii, best_sum, _ = solve_lp(best_centers)

    # Final strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_centers[i,0] - best_centers[j,0], 
                         best_centers[i,1] - best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
