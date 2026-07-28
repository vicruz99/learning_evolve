# sol_000261 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000247 (state 93496474) state=3351f124 sum of radii=2.411726 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    n = centers.shape[0]
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(l, 1e-9)) for l in lims]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A = np.zeros((m, n))
    A[np.arange(m), idx_i] = 1.0
    A[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b = dists[idx_i, idx_j]
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def constraints_joints(x, n):
    cx, cy, r = x[:n], x[n:2*n], x[2*n:]
    con = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    idx = np.triu_indices(n, k=1)
    con = np.concatenate([con, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    return con

def objective_joints(x, n):
    return -np.sum(x[2*n:])

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_c, best_r = None, None
    
    starts = []
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4], [5,5,5,5,6]]
    for pat in patterns:
        pts = []
        r0 = 0.095
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:n])
        starts.append(pts)
        for _ in range(3):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            starts.append(np.clip(p, 0.05, 0.95))

    grid = np.array([(i*0.2+0.1, j*0.2+0.1) for j in range(5) for i in range(5)] + [[0.9, 0.9]])
    starts.append(grid[:n])
    for _ in range(5):
        starts.append(np.clip(grid[:n] + rng.uniform(-0.03, 0.03, (n, 2)), 0.05, 0.95))

    for _ in range(8):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))

    bounds_vars = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
    cons = {'type': 'ineq', 'fun': constraints_joints, 'args': (n,)}

    for cfg in starts:
        r_init = np.full(n, 0.08)
        x0 = np.concatenate([cfg[:,0], cfg[:,1], r_init])
        try:
            res = minimize(objective_joints, x0, args=(n,), method='SLSQP',
                           bounds=bounds_vars, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                r_lp, s_lp = solve_lp_radii(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c = c_opt.copy()
                    best_r = r_lp.copy()
        except Exception:
            pass

    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_sum
        step = 0.015
        
        for it in range(2000):
            idx = rng.integers(n)
            old = curr_c[idx].copy()
            move = rng.uniform(-step, step, 2)
            curr_c[idx] = np.clip(old + move, 1e-4, 1.0-1e-4)
            
            r_try, s_try = solve_lp_radii(curr_c)
            if s_try > curr_s + 1e-9:
                curr_s = s_try
                curr_r = r_try.copy()
                if s_try > best_sum:
                    best_sum = s_try
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step = min(step * 1.05, 0.03)
            else:
                curr_c[idx] = old
                step *= 0.995
                
        for _ in range(100):
            pert_c = best_c.copy()
            k = rng.integers(2, 6)
            idxs = rng.choice(n, k, replace=False)
            pert_c[idxs] += rng.uniform(-0.01, 0.01, (k, 2))
            pert_c = np.clip(pert_c, 0.05, 0.95)
            
            r_p, s_p = solve_lp_radii(pert_c)
            if s_p > best_sum:
                best_sum = s_p
                best_c = pert_c.copy()
                best_r = r_p.copy()
                
    if best_c is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_c[i,0], best_c[i,1], best_r[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                rs = best_r[i] + best_r[j]
                if rs > 1e-12:
                    scale = min(scale, d/rs)
        best_r *= scale * 0.9999995
        best_sum = float(np.sum(best_r))
    else:
        best_c = rng.uniform(0.1, 0.9, (n, 2))
        best_r = np.full(n, 0.05)
        best_sum = float(np.sum(best_r))
        
    return best_c, best_r, best_sum
