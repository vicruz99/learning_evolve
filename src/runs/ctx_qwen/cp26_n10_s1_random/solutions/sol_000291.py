# sol_000291 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000274 (state 2a84d47a) state=d6137685 sum of radii=1.907075 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIU_I, TRIU_J = np.triu_indices(N_CIRCLES, k=1)
N_PAIRS = len(TRIU_I)

def compute_dists(c):
    """Computes pairwise Euclidean distance matrix."""
    dx = c[:, 0, None] - c[:, 0]
    dy = c[:, 1, None] - c[:, 1]
    return np.sqrt(dx**2 + dy**2)

def solve_lp_and_grad(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns objective (negative sum) and its gradient w.r.t centers.
    """
    n = centers.shape[0]
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, lim) for lim in lims]
    
    dists = compute_dists(centers)
    np.fill_diagonal(dists, np.inf)
    b_ub = dists[TRIU_I, TRIU_J]
    
    # Construct constant structure inequality matrix
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), TRIU_I] = 1.0
    A_ub[np.arange(N_PAIRS), TRIU_J] = 1.0
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return 1e6, np.zeros(n * 2)
            
        r = res.x
        obj = -np.sum(r)
        grad = np.zeros_like(centers)
        
        # Extract dual variables (marginals) for pairwise constraints
        if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            lams = np.abs(res.ineqlin.marginals)
            active = lams > 1e-7
            if np.any(active):
                k = np.where(active)[0]
                lam_k = lams[k]
                ii = TRIU_I[k]
                jj = TRIU_J[k]
                d_ij = dists[ii, jj]
                d_safe = np.where(d_ij > 1e-9, d_ij, 1.0)
                diff = centers[ii] - centers[jj]
                factors = (lam_k / d_safe)[:, None]
                
                np.add.at(grad, ii, diff * factors)
                np.add.at(grad, jj, -diff * factors)
                
        # We minimize negative sum, so gradient of obj is -gradient of sum
        return obj, -grad.flatten()
    except Exception:
        return 1e6, np.zeros(n * 2)

def generate_hex(row_counts, r0):
    """Generates a hexagonal lattice configuration with specified row counts."""
    n = N_CIRCLES
    pts = []
    y = r0
    for i, cnt in enumerate(row_counts):
        shift = r0 if i % 2 == 1 else 0.0
        width = (cnt - 1) * 2 * r0
        x_start = 0.5 - width / 2.0 + shift
        for k in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x_start + k * 2 * r0, y])
        y += np.sqrt(3) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [6,5,4,6,5], [5,4,6,6,5], [7,5,5,5,4],
        [5,6,6,5,4], [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6]
    ]
    
    starts = []
    for pat in patterns:
        if sum(pat) != n: continue
        for r0 in [0.085, 0.095, 0.105]:
            pts = generate_hex(pat, r0)
            starts.append(pts)
            # Controlled perturbations to break symmetry
            p = pts + rng.uniform(-0.025, 0.025, pts.shape)
            p = np.clip(p, 0.05, 0.95)
            starts.append(p)
            
    # Add random dense starts
    for _ in range(10):
        starts.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds_opt = [(0.02, 0.98)] * (2 * n)
    
    # Phase 1: Gradient-based optimization from diverse starts
    for cfg in starts:
        x0 = cfg.flatten()
        try:
            res = minimize(solve_lp_and_grad, x0, method='L-BFGS-B', bounds=bounds_opt,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(n, 2)
                
                lims = np.minimum(np.minimum(c_opt[:, 0], 1.0 - c_opt[:, 0]),
                                  np.minimum(c_opt[:, 1], 1.0 - c_opt[:, 1]))
                dists = compute_dists(c_opt)
                np.fill_diagonal(dists, np.inf)
                b_ub = dists[TRIU_I, TRIU_J]
                A_ub = np.zeros((N_PAIRS, n))
                A_ub[np.arange(N_PAIRS), TRIU_I] = 1.0
                A_ub[np.arange(N_PAIRS), TRIU_J] = 1.0
                bounds_lp = [(0.0, max(l, 1e-9)) for l in lims]
                
                res_lp = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
                if res_lp.success:
                    r_opt = res_lp.x
                    s_opt = np.sum(r_opt)
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative hill climbing refinement
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        A_ub = np.zeros((N_PAIRS, n))
        A_ub[np.arange(N_PAIRS), TRIU_I] = 1.0
        A_ub[np.arange(N_PAIRS), TRIU_J] = 1.0
        
        for step in range(4000):
            scale = 0.008 * (1.0 - step/4000.0)**0.4
            i = rng.integers(n)
            old = curr_c[i].copy()
            curr_c[i] += rng.uniform(-scale, scale, 2)
            curr_c[i] = np.clip(curr_c[i], 0.03, 0.97)
            
            lims = np.minimum(np.minimum(curr_c[:, 0], 1.0 - curr_c[:, 0]),
                              np.minimum(curr_c[:, 1], 1.0 - curr_c[:, 1]))
            dists = compute_dists(curr_c)
            np.fill_diagonal(dists, np.inf)
            b_ub = dists[TRIU_I, TRIU_J]
            bounds_lp = [(0.0, max(l, 1e-9)) for l in lims]
            
            try:
                res_lp = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
                if res_lp.success:
                    s_new = np.sum(res_lp.x)
                    if s_new > curr_s + 1e-9:
                        curr_s = s_new
                        curr_r = res_lp.x.copy()
                    else:
                        curr_c[i] = old
                else:
                    curr_c[i] = old
            except Exception:
                curr_c[i] = old
                
        best_centers = curr_c
        best_radii = curr_r
        best_sum = curr_s

    # Fallback configuration
    if best_centers is None:
        best_centers = generate_hex([6,5,6,5,4], 0.09)
        lims = np.minimum(np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]),
                          np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1]))
        dists = compute_dists(best_centers)
        np.fill_diagonal(dists, np.inf)
        b_ub = dists[TRIU_I, TRIU_J]
        A_ub = np.zeros((N_PAIRS, n))
        A_ub[np.arange(N_PAIRS), TRIU_I] = 1.0
        A_ub[np.arange(N_PAIRS), TRIU_J] = 1.0
        bounds_lp = [(0.0, max(l, 1e-9)) for l in lims]
        res_lp = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        best_radii = res_lp.x if res_lp.success else np.full(n, 0.05)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict numerical safety scaling
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    dists = compute_dists(best_centers)
    r_pair = best_radii[:, None] + best_radii[None, :]
    dists_safe = np.where(dists > 1e-9, dists, 1.0)
    scale = min(scale, np.min(dists_safe[TRIU_I, TRIU_J] / np.maximum(r_pair[TRIU_I, TRIU_J], 1e-12)))
    
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
