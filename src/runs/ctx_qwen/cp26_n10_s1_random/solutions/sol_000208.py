# sol_000208 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000165 (state ab534a56) state=9cb32c70 sum of radii=2.231440 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def compute_min_clearance(centers):
    """Computes the maximum feasible equal radius for given centers."""
    d_bound = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    min_bound = np.min(d_bound)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(min_bound, min_pairwise)

def neg_clearance(x):
    """Objective for Nelder-Mead: maximize minimum clearance."""
    centers = x.reshape(N_CIRCLES, 2)
    return -compute_min_clearance(centers)

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    A_ub = []
    b_ub = []
    
    # Boundary constraints for radii
    for i in range(n):
        x, y = centers[i]
        mx = max(0.0, min(x, 1.0 - x, y, 1.0 - y))
        bounds.append((0.0, mx))
        for lim in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def hex_init(row_counts, r_guess=0.10):
    """Generates a centered hexagonal lattice configuration."""
    pts = []
    y = r_guess
    for i, cnt in enumerate(row_counts):
        shift = r_guess if i % 2 == 1 else 0.0
        x = r_guess + shift
        for _ in range(cnt):
            pts.append([x, y])
            x += 2.0 * r_guess
        y += np.sqrt(3) * r_guess
    while len(pts) < N_CIRCLES:
        pts.append([0.5, 0.5])
    return np.array(pts[:N_CIRCLES])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    rng = np.random.default_rng(42)
    
    # Diverse row distributions known to be near-optimal for n=26
    row_configs = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 5, 5, 6, 4],
        [5, 6, 6, 4, 5], [5, 5, 5, 5, 6], [7, 6, 6, 7],
        [6, 7, 6, 7], [5, 6, 5, 6, 5, 1], [4, 5, 6, 6, 5]
    ]
    
    best_centers = None
    best_clearance = -1.0
    
    # Phase 1: Optimize centers to maximize minimum clearance
    for rc in row_configs:
        if sum(rc) < n:
            continue
        base = hex_init(rc, r_guess=0.095)
        
        # Center and scale to fit tightly within the square initially
        mn = base.min(axis=0)
        mx = base.max(axis=0)
        span = mx - mn
        if np.max(span) > 1e-6:
            scale = 0.80 / np.max(span)
            base = (base - mn) * scale + (1.0 - 0.80) / 2.0
            
        starts = [base]
        for _ in range(3):
            pert = base + rng.uniform(-0.03, 0.03, (n, 2))
            starts.append(np.clip(pert, 0.05, 0.95))
            
        for cfg in starts:
            x0 = cfg.flatten()
            try:
                res = minimize(neg_clearance, x0, method='Nelder-Mead', 
                              options={'maxiter': 6000, 'xatol': 1e-7, 'fatol': 1e-9})
                centers_opt = res.x.reshape(n, 2)
                clr = -res.fun
                if clr > best_clearance:
                    best_clearance = clr
                    best_centers = centers_opt.copy()
            except Exception:
                continue
                
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = hex_init([6, 5, 6, 5, 4], 0.095)
        
    # Phase 2: LP refinement to extract maximal sum of radii for the best layout
    radii_lp = solve_lp_radii(best_centers)
    if radii_lp is not None:
        best_radii = radii_lp
        best_sum = np.sum(best_radii)
    else:
        best_radii = np.full(n, best_clearance)
        best_sum = n * best_clearance
        
    # Phase 3: Local coordinate descent refinement using LP objective
    # Move one circle at a time to random nearby positions, keep if LP sum improves
    for _ in range(150):
        idx = rng.integers(n)
        old_pos = best_centers[idx].copy()
        cand = best_centers[idx] + rng.uniform(-0.015, 0.015, 2)
        cand = np.clip(cand, 0.01, 0.99)
        best_centers[idx] = cand
        
        r_new = solve_lp_radii(best_centers)
        if r_new is not None:
            s_new = np.sum(r_new)
            if s_new > best_sum + 1e-7:
                best_sum = s_new
                best_radii = r_new.copy()
            else:
                best_centers[idx] = old_pos
        else:
            best_centers[idx] = old_pos
            
    # Final safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
