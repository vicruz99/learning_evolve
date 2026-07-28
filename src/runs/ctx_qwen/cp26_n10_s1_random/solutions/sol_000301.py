# sol_000301 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000289 (state 50cd4c97) state=e1bd7b36 sum of radii=2.541270 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(TRIU_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = N
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, 1e9)
    
    A = np.zeros((NUM_PAIRS, n))
    A[np.arange(NUM_PAIRS), TRIU_I] = 1.0
    A[np.arange(NUM_PAIRS), TRIU_J] = 1.0
    b = dists[TRIU_I, TRIU_J]
    
    bounds = [(0.0, lims[i]) for i in range(n)]
    c_obj = -np.ones(n)
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-7):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-7), 0.0

def obj_slqp(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_slqp(v):
    """Inequality constraints >= 0 using squared distances for smooth gradients."""
    cx = v[:N]
    cy = v[N:2 * N]
    r = v[2 * N:]
    
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    dx = cx[TRIU_I] - cx[TRIU_J]
    dy = cy[TRIU_I] - cy[TRIU_J]
    dr = r[TRIU_I] + r[TRIU_J]
    
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def run_packing():
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_c = None
    best_r = None
    
    starts = []
    pats = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4],
        [7, 5, 5, 5, 4], [4, 5, 7, 5, 5], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [5, 7, 5, 5, 4]
    ]
    
    for pat in pats:
        if sum(pat) != N:
            continue
        pts = []
        y = 0.095
        for i, cnt in enumerate(pat):
            shift = 0.095 if i % 2 == 1 else 0.0
            x = 0.095 + shift
            for _ in range(cnt):
                if len(pts) >= N:
                    break
                pts.append([x, y])
                x += 0.19
            y += 0.095 * np.sqrt(3)
        starts.append(np.array(pts[:N]))
        
    for _ in range(4):
        g = np.linspace(0.12, 0.88, 5)
        grid = np.array([(x, y) for y in g for x in g])
        grid = np.vstack([grid, [0.5, 0.5]])
        starts.append(grid + rng.uniform(-0.02, 0.02, grid.shape))
        
    for _ in range(8):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    for s in starts:
        # Phase 1: Force-directed simulation to densify and escape local minima
        c = s.copy()
        r = np.full(N, 0.03)
        vel = np.zeros_like(c)
        
        for step in range(1200):
            dt = 0.008 * (1.0 - 0.8 * step / 1200) + 0.0005
            damp = 0.85
            k = 200.0
            r *= 1.00012
            
            diff = c[:, None, :] - c[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            np.fill_diagonal(dist, 1e9)
            
            overlap = np.maximum(0.0, r[:, None] + r[None, :] - dist)
            inv_d = 1.0 / (dist + 1e-9)
            rep_mag = overlap * inv_d * k
            f_rep = np.sum(diff * rep_mag[:, :, None], axis=1)
            
            f_wall = np.zeros_like(c)
            f_wall[:, 0] += np.clip(r - c[:, 0], 0, None) * k
            f_wall[:, 0] -= np.clip(c[:, 0] + r - 1.0, 0, None) * k
            f_wall[:, 1] += np.clip(r - c[:, 1], 0, None) * k
            f_wall[:, 1] -= np.clip(c[:, 1] + r - 1.0, 0, None) * k
            
            vel = vel * damp + (f_rep + f_wall) * dt
            c += vel
            c = np.clip(c, 1e-4, 1.0 - 1e-4)
            
        # Phase 2: SLSQP Joint Polish
        r_init, _ = solve_lp(c)
        r_init = r_init * 0.96
        v0 = np.concatenate([c[:, 0], c[:, 1], r_init])
        bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
        
        try:
            res = minimize(obj_slqp, v0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': cons_slqp},
                           options={'maxiter': 4000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_pol = np.column_stack((res.x[:N], res.x[N:2 * N]))
                r_lp, s_lp = solve_lp(c_pol)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c = c_pol.copy()
                    best_r = r_lp.copy()
        except Exception:
            pass
            
    # Phase 3: Simulated Annealing on centers with exact LP radii evaluation
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_sum
        T = 0.004
        step = 0.012
        
        for it in range(6000):
            i = rng.integers(N)
            old = curr_c[i].copy()
            curr_c[i] += rng.normal(0, step, 2)
            curr_c[i] = np.clip(curr_c[i], 1e-4, 0.999)
            
            r_try, s_try = solve_lp(curr_c)
            delta = s_try - curr_s
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                curr_s = s_try
                curr_r = r_try.copy()
                T *= 0.9992
                step *= 0.9996
            else:
                curr_c[i] = old
                
        best_c = curr_c
        best_r = curr_r
        best_sum = curr_s
        
    # Phase 4: Strict numerical safety scaling
    scale = 1.0
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    diff = best_c[:, None, :] - best_c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_r *= scale * 0.999999
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
