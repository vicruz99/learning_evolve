# sol_000294 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000288 (state 4522f7fa) state=e7695cc0 sum of radii=2.284290 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N_CIRCLES = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    
    idx_i, idx_j = np.triu_indices(n, 1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def obj_lp_sum(x_flat):
    """Objective for L-BFGS-B: maximize LP sum of radii."""
    c = np.clip(x_flat.reshape(N_CIRCLES, 2), 1e-4, 1.0 - 1e-4)
    _, s = solve_lp(c)
    return -s

def get_max_equal_radius(c):
    """Computes the maximum feasible equal radius D for given centers."""
    c = np.clip(c, 1e-4, 1.0 - 1e-4)
    wall = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), np.minimum(c[:, 1], 1.0 - c[:, 1]))
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    return np.min(np.concatenate([wall, 0.5 * np.min(dists)]))

def obj_equal_d(x_flat):
    """Objective for L-BFGS-B: maximize equal radius D."""
    return -get_max_equal_radius(x_flat.reshape(N_CIRCLES, 2))

def generate_starts(rng):
    """Generates diverse high-quality initial configurations."""
    starts = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 6, 6, 4], [6, 6, 4, 6, 4],
        [5, 5, 6, 5, 5], [6, 4, 6, 4, 6], [5, 6, 6, 5, 4], [7, 5, 5, 5, 4],
        [5, 7, 5, 5, 4], [4, 5, 7, 5, 5], [5, 5, 7, 5, 4], [6, 6, 6, 4, 4]
    ]
    
    for pat in patterns:
        if sum(pat) != N_CIRCLES:
            continue
        pts = []
        r0 = 0.10
        y = r0
        for row_idx, cnt in enumerate(pat):
            shift = r0 if row_idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N_CIRCLES:
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
            
        arr = np.array(pts[:N_CIRCLES])
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        span = mx - mn
        if span[0] > 1e-4 and span[1] > 1e-4:
            arr = (arr - mn) / span * 0.8 + 0.1
        starts.append(arr)
        
        for _ in range(2):
            p = arr + rng.uniform(-0.02, 0.02, arr.shape)
            starts.append(np.clip(p, 0.02, 0.98))
            
    for _ in range(4):
        starts.append(rng.uniform(0.15, 0.85, (N_CIRCLES, 2)))
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_c = None
    best_r = None
    
    starts = generate_starts(rng)
    bounds_opt = [(0.01, 0.99)] * (2 * N_CIRCLES)
    
    # Phase 1: Optimize equal radius layout to find optimal structural baseline
    for cfg in starts:
        x0 = cfg.flatten()
        try:
            res = minimize(obj_equal_d, x0, method='L-BFGS-B', bounds=bounds_opt,
                           options={'maxiter': 800, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N_CIRCLES, 2)
                
                # Phase 2: Optimize unequal radii sum from the structured layout
                res2 = minimize(obj_lp_sum, c_opt.flatten(), method='L-BFGS-B', bounds=bounds_opt,
                                options={'maxiter': 1200, 'ftol': 1e-13, 'disp': False})
                if np.isfinite(res2.fun):
                    c_final = res2.x.reshape(N_CIRCLES, 2)
                    r_final, s_final = solve_lp(c_final)
                    if s_final > best_sum:
                        best_sum = s_final
                        best_c = c_final.copy()
                        best_r = r_final.copy()
        except Exception:
            pass
            
    # Fallback if optimization fails
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum = solve_lp(best_c)
        
    # Phase 3: Coordinate-wise hill-climbing for fine-tuning
    curr_c = best_c.copy()
    curr_r, curr_s = solve_lp(curr_c)
    best_sum = curr_s
    best_c = curr_c.copy()
    best_r = curr_r.copy()
    
    step = 0.012
    for sweep in range(8):
        improved = False
        for i in range(N_CIRCLES):
            old = curr_c[i].copy()
            best_move = None
            best_move_s = curr_s
            for _ in range(6):
                move = rng.uniform(-step, step, 2)
                curr_c[i] = np.clip(old + move, 0.01, 0.99)
                r_try, s_try = solve_lp(curr_c)
                if s_try > best_move_s:
                    best_move_s = s_try
                    best_move = curr_c[i].copy()
                else:
                    curr_c[i] = old
            if best_move is not None:
                curr_c[i] = best_move
                curr_r, curr_s = solve_lp(curr_c)
                if curr_s > best_sum + 1e-8:
                    best_sum = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                    improved = True
            else:
                curr_c[i] = old
        if not improved:
            break
        step *= 0.85
        
    # Phase 4: Strict numerical safety scaling
    scale = 1.0
    for i in range(N_CIRCLES):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_r *= scale * 0.999999
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
