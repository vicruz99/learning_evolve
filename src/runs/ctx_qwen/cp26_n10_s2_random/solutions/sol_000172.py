# sol_000172 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000147 (state da2cd853) state=8b7860db sum of radii=2.583307 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
n_pairs = N * (N - 1) // 2
n_bound = 4 * N
m = n_pairs + n_bound

# Precompute LP constraint matrix structure (constant for fixed N)
A_ub_pre = np.zeros((m, N))
pair_indices = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_ub_pre[idx, i] = 1.0
        A_ub_pre[idx, j] = 1.0
        pair_indices.append((i, j))
        idx += 1
for i in range(N):
    for _ in range(4):
        A_ub_pre[idx, i] = 1.0
        idx += 1

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes gradient via duals."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(m)
    idx = 0
    for i, j in pair_indices:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    # Maximize sum(r) => Minimize -sum(r)
    res = linprog(-np.ones(N), A_ub=A_ub_pre, b_ub=b_ub, bounds=[(0, None)]*N, method='highs')
    if not res.success:
        return np.full(N, 1e-7), 0.0, np.zeros_like(centers)
        
    radii = res.x
    sum_r = np.sum(radii)
    duals = res.ineqlin.marginals
    
    # Gradient of sum(r) w.r.t centers using LP duals
    grad = np.zeros_like(centers)
    idx = 0
    for i, j in pair_indices:
        lam = duals[idx]
        if lam > 1e-10:
            d = dists[i, j]
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    b_start = n_pairs
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, sum_r, grad

def obj_func(v):
    return -solve_lp_and_grad(v.reshape(N, 2))[1]

def grad_func(v):
    return -solve_lp_and_grad(v.reshape(N, 2))[2].flatten()

def generate_starts(rng):
    """Generates diverse hexagonal lattice starts with rotations and scales."""
    starts = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 5, 5, 5, 5], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6]
    ]
    for pat in patterns:
        for r_est in [0.095, 0.10, 0.105]:
            for rot in [0.0, 0.15, -0.15]:
                c = []
                y = r_est
                for r_idx, cnt in enumerate(pat):
                    shift = r_est if r_idx % 2 == 1 else 0.0
                    x = r_est + shift
                    for _ in range(cnt):
                        if len(c) < N: c.append([x, y])
                        x += 2.0 * r_est
                    y += r_est * np.sqrt(3)
                c = np.array(c[:N])
                if rot != 0.0:
                    c = c @ np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                c -= c.min(axis=0)
                scale = c.max(axis=0) - c.min(axis=0)
                scale = np.max(scale)
                if scale > 1e-9:
                    c = c / scale * 0.92 + 0.04
                c += rng.normal(0, 0.001, c.shape)
                c = np.clip(c, 0.01, 0.99)
                starts.append(c)
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c, best_sum = None, -1.0
    bounds = [(0.005, 0.995)] * (2 * N)
    starts = generate_starts(rng)
    
    # Phase 1: Gradient Ascent with L-BFGS-B
    for c0 in starts:
        try:
            res = minimize(obj_func, c0.flatten(), jac=grad_func, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 400, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
        except: pass
        
    if best_c is None:
        best_c = starts[0]
        _, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Simulated Annealing to escape local minima
    curr_c, curr_s = best_c.copy(), best_sum
    T = 0.004
    for _ in range(1000):
        T *= 0.999
        nc = np.clip(curr_c + rng.normal(0, T, curr_c.shape), 0.01, 0.99)
        _, ns, _ = solve_lp_and_grad(nc)
        if ns > curr_s or rng.random() < np.exp((ns - curr_s)/max(T*2.0, 1e-6)):
            curr_c, curr_s = nc, ns
            if curr_s > best_sum:
                best_sum = curr_s
                best_c = curr_c.copy()
                
    # Phase 3: Joint SLSQP Polish
    def obj_joint(v): return -np.sum(v[2*N:])
    def cons_joint(v):
        c, r = v[:2*N].reshape(N, 2), v[2*N:]
        con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
        d = np.linalg.norm(c[:,None,:] - c[None,:,:], axis=2)
        con.append(d[np.triu_indices(N,1)] - (r[:,None]+r[None,:])[np.triu_indices(N,1)])
        return np.concatenate(con)
        
    v0 = np.concatenate([best_c.flatten(), solve_lp_and_grad(best_c)[0]])
    bounds_j = [(0.005, 0.995)]*(2*N) + [(0.0, 0.5)]*N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_j,
                       constraints={'type':'ineq', 'fun':cons_joint},
                       options={'maxiter':8000, 'ftol':1e-13})
        if -res.fun > best_sum - 1e-8:
            best_c = res.x[:2*N].reshape(N, 2)
            best_r = res.x[2*N:]
            best_sum = -res.fun
        else:
            best_r, _, _ = solve_lp_and_grad(best_c)
    except:
        best_r, _, _ = solve_lp_and_grad(best_c)
        
    # Phase 4: Strict Deterministic Repair
    centers, radii = best_c.copy(), best_r.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr - 1e-11
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d - 1e-11:
                    sh = (radii[i] + radii[j] - d)*0.5 + 1e-11
                    radii[i] -= sh; radii[j] -= sh
                    changed = True
        if not changed: break
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
