# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000034 (state 766fe0af) state=68726102 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
# Precompute upper triangle indices for pairwise constraints
TRI_U_IDX = np.triu_indices(N, k=1)

def compute_constraints(vars_flat):
    """Computes all boundary and non-overlap constraints. Returns array >= 0."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]
    
    c = []
    # Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Overlap: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    dist_sq = dx**2 + dy**2
    dr_sq = dr**2
    
    c.append(dist_sq[TRI_U_IDX] - dr_sq[TRI_U_IDX])
    return np.concatenate(c)

def compute_objective(vars_flat):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_flat[2::3])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]"""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def make_hex_init(row_counts, shift_pattern, r_scale=0.095, noise=0.0):
    """Generates a feasible hexagonal lattice configuration."""
    pts = []
    y = r_scale
    dy = r_scale * np.sqrt(3.0)
    dx = 2.0 * r_scale
    
    for r_idx, count in enumerate(row_counts):
        shift = shift_pattern[r_idx] * (dx / 2.0)
        x = r_scale + shift
        for _ in range(count):
            pts.append([x, y])
            x += dx
        y += dy
        
    pts = np.array(pts[:N])
    if noise > 0:
        rng = np.random.default_rng(42)
        pts += rng.normal(0, noise, pts.shape)
    pts = np.clip(pts, 0.0, 1.0)
    
    rs = np.full(N, r_scale * 0.8)
    x0 = np.zeros(N * 3)
    for i in range(N):
        x0[3*i] = pts[i, 0]
        x0[3*i+1] = pts[i, 1]
        x0[3*i+2] = rs[i]
    return x0

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Pairwise: r_i + r_j <= dist(i, j)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 2.0) # Ignore self
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        bounds = [centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1]]
        for b in bounds:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                  bounds=(0.0, None), method='highs')
    if res.success:
        return res.x
    return np.full(n, 0.01)

def run_packing() -> tuple:
    np.random.seed(123)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_val = -np.inf
    best_x = None
    
    # Phase 1: Multi-start from diverse hexagonal patterns
    configs = [
        ([5, 5, 5, 5, 5, 1], [0, 1, 0, 1, 0, 0]),
        ([5, 6, 5, 6, 4], [0, 1, 0, 1, 0]),
        ([6, 5, 6, 5, 4], [0, 1, 0, 1, 0]),
        ([5, 5, 6, 5, 5], [0, 1, 0, 1, 0]),
        ([4, 6, 6, 6, 4], [0, 1, 0, 1, 0]),
        ([6, 6, 5, 5, 4], [0, 1, 0, 1, 0]),
    ]
    
    initial_candidates = []
    for rc, sp in configs:
        initial_candidates.append(make_hex_init(rc, sp, r_scale=0.095, noise=0.002))
        initial_candidates.append(make_hex_init(rc, sp, r_scale=0.090, noise=0.005))
        
    # Add random starts
    for _ in range(8):
        rng = np.random.default_rng(_)
        centers = rng.uniform(0.15, 0.85, (N, 2))
        rs = np.full(N, 0.04)
        x0 = np.zeros(N*3)
        for i in range(N):
            x0[3*i] = centers[i,0]
            x0[3*i+1] = centers[i,1]
            x0[3*i+2] = rs[i]
        initial_candidates.append(x0)
        
    for x0 in initial_candidates:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
            if np.all(compute_constraints(res.x) >= -1e-5):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        best_x = initial_candidates[0]
        
    # Phase 2: Iterative Radius Boosting
    curr_x = best_x.copy()
    for step in range(25):
        rs = curr_x[2::3].copy()
        rs *= 1.0015
        curr_x[2::3] = rs
        
        # Perturb centers to help resolve new overlaps
        curr_x[:2*N] += np.random.normal(0, 0.0004, 2*N)
        for i in range(N):
            r = curr_x[3*i+2]
            curr_x[3*i] = np.clip(curr_x[3*i], r, 1.0-r)
            curr_x[3*i+1] = np.clip(curr_x[3*i+1], r, 1.0-r)
            
        try:
            res = minimize(compute_objective, curr_x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-13})
            if np.all(compute_constraints(res.x) >= -1e-5):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
                    curr_x = best_x.copy()
        except Exception:
            pass
            
    # Phase 3: Exact LP refinement for radii given optimal centers
    centers_opt = best_x.reshape(N, 3)[:, :2]
    radii_lp = solve_lp_radii(centers_opt)
    lp_sum = np.sum(radii_lp)
    
    if lp_sum > best_val:
        best_x[2::3] = radii_lp
        best_val = lp_sum
        
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x.reshape(N, 3)[:, 2]
    
    # Final safety clamp & overlap resolution
    for _ in range(10):
        changed = False
        for i in range(N):
            x, y, r = centers[i,0], centers[i,1], radii[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
                
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-14
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    return centers, radii, final_sum
