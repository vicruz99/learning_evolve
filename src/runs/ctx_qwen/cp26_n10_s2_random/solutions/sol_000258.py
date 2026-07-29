# sol_000258 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000205 (state 0b4dbf91) state=70ffb139 sum of radii=2.293681 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
A_LP = None
PAIR_IDX = None

def setup_lp():
    global A_LP, PAIR_IDX
    num_pairs = N * (N - 1) // 2
    A_LP = np.zeros((num_pairs + 4 * N, N))
    PAIR_IDX = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_LP[k, i] = 1.0
            A_LP[k, j] = 1.0
            PAIR_IDX.append((i, j))
            k += 1
    for i in range(N):
        for _ in range(4):
            A_LP[k, i] = 1.0
            k += 1

setup_lp()

def solve_lp_and_grad(centers):
    c = np.clip(centers, 1e-8, 1.0 - 1e-8)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = c[i, 0]; k += 1
        b[k] = 1.0 - c[i, 0]; k += 1
        b[k] = c[i, 1]; k += 1
        b[k] = 1.0 - c[i, 1]; k += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    duals = np.zeros(b.shape[0])
    if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
        duals = np.asarray(res.marginals.ineqlin)
    elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
        duals = np.asarray(res.ineqlin.marginals)
        
    grad = np.zeros_like(c)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-8:
                vec = (c[i] - c[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4 * i] - duals[bound_start + 4 * i + 1]
        grad[i, 1] += duals[bound_start + 4 * i + 2] - duals[bound_start + 4 * i + 3]
        
    return radii, np.sum(radii), grad

def gradient_ascent(c0, steps=1500, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    c = c0.copy()
    best_c = c.copy()
    best_s = -1.0
    step = 0.006
    
    r, s, g = solve_lp_and_grad(c)
    best_s = s
    
    for k in range(steps):
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        direction = g / gn
        c_new = c + step * direction
        c_new = np.clip(c_new, 0.002, 0.998)
        
        r_new, s_new, g_new = solve_lp_and_grad(c_new)
        if s_new > s + 1e-10:
            c = c_new
            s = s_new
            g = g_new
            step = min(step * 1.04, 0.025)
            if s > best_s:
                best_s = s
                best_c = c.copy()
        else:
            step *= 0.88
            if step < 1e-9:
                break
                
        if k > 0 and k % 120 == 0:
            noise = 0.003 * (0.6 ** (k // 240))
            c += rng.normal(0, noise, c.shape)
            c = np.clip(c, 0.002, 0.998)
            r, s, g = solve_lp_and_grad(c)
            
    return best_c, best_s

def slsqp_polish(c0, r0):
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(v):
        return -np.sum(v[2 * N:])
        
    def cons(v):
        cx = v[:2 * N].reshape(N, 2)
        rx = v[2 * N:]
        c_list = [cx[:, 0] - rx, 1.0 - cx[:, 0] - rx, 
                  cx[:, 1] - rx, 1.0 - cx[:, 1] - rx]
        i_idx, j_idx = np.triu_indices(N, 1)
        dx = cx[i_idx, 0] - cx[j_idx, 0]
        dy = cx[i_idx, 1] - cx[j_idx, 1]
        dr = rx[i_idx] + rx[j_idx]
        c_list.append(dx**2 + dy**2 - dr**2)
        return np.concatenate(c_list)
        
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        if res.success and np.min(cons(res.x)) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], np.sum(res.x[2 * N:])
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_starts(n, rng):
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 6, 4, 5, 6], [5, 5, 4, 6, 6], [6, 6, 4, 5, 5],
        [5, 7, 5, 5, 4], [4, 6, 5, 6, 5], [5, 6, 6, 4, 5],
        [6, 5, 5, 6, 4], [5, 5, 6, 6, 4], [4, 5, 5, 6, 6],
        [6, 6, 5, 4, 5], [5, 6, 5, 5, 5], [6, 5, 6, 4, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.090, 0.095, 0.100, 0.105]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < n:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:n])
            c += rng.normal(0, 0.0025, c.shape)
            c = np.clip(c, 0.03, 0.97)
            starts.append(c)
            
    for _ in range(15):
        starts.append(rng.uniform(0.12, 0.88, (n, 2)))
        
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (n, 2))
        c[:4] = [[0.07, 0.07], [0.93, 0.07], [0.07, 0.93], [0.93, 0.93]]
        starts.append(c)
        
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (n, 2))
        for _ in range(400):
            f = np.zeros_like(c)
            for i in range(n):
                for j in range(i + 1, n):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-4:
                        push = (0.22 - d) / d
                        f[i] += d_vec * push
                        f[j] -= d_vec * push
            c += f * 0.003
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
    return starts

def repair_packing(centers, radii):
    radii = radii.copy()
    for _ in range(80):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0],
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0],
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-11:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_s = -1.0
    best_r = None
    
    starts = generate_starts(N, rng)
    
    candidates = []
    for c0 in starts:
        c_opt, s_opt = gradient_ascent(c0, steps=1800, rng=rng)
        candidates.append((s_opt, c_opt))
        
    candidates.sort(reverse=True)
    
    for s_val, c_val in candidates[:15]:
        if s_val <= best_s:
            break
        r_val, _, _ = solve_lp_and_grad(c_val)
        c_pol, r_pol, s_pol = slsqp_polish(c_val, r_val)
        if s_pol > best_s:
            best_s = s_pol
            best_c = c_pol
            best_r = r_pol
            
    centers = best_c.copy()
    radii = repair_packing(centers, best_r.copy())
    
    return centers, radii, float(np.sum(radii))
