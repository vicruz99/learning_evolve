# sol_000236 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000142 (state d65765d5) state=402f1f06 sum of radii=2.022675 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def get_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Distance to boundaries limits maximum possible radius for each circle
    lims = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    bounds = [(0.0, max(l, 1e-9)) for l in lims]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    m = len(i_idx)
    A = np.zeros((m, n))
    A[np.arange(m), i_idx] = 1.0
    A[np.arange(m), j_idx] = 1.0
    
    diff = centers[i_idx] - centers[j_idx]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    b = dists
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-7)

def slsqp_objective(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def slsqp_constraints(v, n):
    """Inequality constraints >= 0 for valid packing."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = []
    # Boundary constraints
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    x_diff = x[:, np.newaxis] - x[np.newaxis, :]
    y_diff = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(x_diff**2 + y_diff**2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    c.append(dists[i_idx, j_idx] - r_sum[i_idx, j_idx])
    
    return np.concatenate(c)

def generate_hex_starts(n, rng):
    """Generates diverse hexagonal lattice initial configurations."""
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [6, 6, 4, 6, 4], [5, 5, 6, 5, 5], [6, 4, 6, 4, 6],
        [7, 5, 7, 5], [5, 7, 5, 7], [6, 5, 7, 5], [5, 6, 5, 7],
        [8, 5, 6, 5], [5, 8, 5, 6], [6, 6, 6, 4, 4], [4, 4, 6, 6, 6]
    ]
    
    starts = []
    for pat in patterns:
        if sum(pat) != n:
            continue
        pts = []
        r0 = 0.095
        y = r0
        for row_idx, cnt in enumerate(pat):
            shift = r0 if row_idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n:
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            
        pts = np.array(pts[:n])
        # Normalize to fit comfortably inside [0.05, 0.95]
        pts = pts * 0.85 + 0.075
        starts.append(pts)
        
        # Add perturbations to break symmetry
        for _ in range(3):
            p = pts + rng.normal(0, 0.025, pts.shape)
            p = np.clip(p, 0.05, 0.95)
            starts.append(p)
            
    # Add fully random starts for diversity
    for _ in range(8):
        starts.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    starts = generate_hex_starts(n, rng)
    bnds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n
    cons = {'type': 'ineq', 'fun': slsqp_constraints, 'args': (n,)}
    
    # Phase 1: Multi-start SLSQP optimization
    for cfg in starts:
        v0 = np.zeros(3 * n)
        v0[0::3] = cfg[:, 0]
        v0[1::3] = cfg[:, 1]
        v0[2::3] = 0.09
        
        try:
            res = minimize(slsqp_objective, v0, args=(n,), method='SLSQP', bounds=bnds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                # LP refinement extracts the mathematically exact max radii for fixed centers
                r_lp = get_lp_radii(c_opt)
                s_lp = np.sum(r_lp)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass
            
    # Phase 2: Local search on centers using LP objective
    if best_centers is not None:
        current_centers = best_centers.copy()
        current_radii = best_radii.copy()
        current_sum = best_sum
        
        step_size = 0.008
        for step in range(3000):
            i = rng.integers(n)
            old_c = current_centers[i].copy()
            
            # Random perturbation
            new_c = old_c + rng.uniform(-step_size, step_size, 2)
            new_c = np.clip(new_c, 0.01, 0.99)
            
            current_centers[i] = new_c
            r_tmp = get_lp_radii(current_centers)
            new_sum = np.sum(r_tmp)
            
            if new_sum > current_sum + 1e-8:
                current_radii = r_tmp
                current_sum = new_sum
                step_size = min(step_size * 1.05, 0.02)
            else:
                current_centers[i] = old_c
                step_size *= 0.98
                
            if step_size < 1e-6:
                step_size = 0.005
                
        if current_sum > best_sum:
            best_centers = current_centers
            best_radii = current_radii
            best_sum = current_sum
            
    # Phase 3: Strict numerical safety scaling to guarantee 1e-12 tolerance
    if best_radii is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if r > 1e-12:
                scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
                
        i_idx, j_idx = np.triu_indices(n, k=1)
        diff = best_centers[i_idx] - best_centers[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        r_pair = best_radii[i_idx] + best_radii[j_idx]
        
        valid_pairs = r_pair > 1e-12
        if np.any(valid_pairs):
            ratios = np.where(valid_pairs, dists / r_pair, 1.0)
            scale = min(scale, np.min(ratios))
            
        best_radii *= scale * 0.9999995
        best_sum = float(np.sum(best_radii))
        
    # Fallback (should not be reached with valid optimization)
    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, (n, 2))
        best_radii = np.full(n, 0.08)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
