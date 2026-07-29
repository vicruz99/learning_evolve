# sol_000217 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000168 (state 79899e79) state=1e190478 sum of radii=2.267915 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_IND = np.triu_indices(N, 1)

def build_lp_structure(n):
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A = np.zeros((num_pairs + num_bound, n))
    pair_idx = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            pair_idx.append((i, j))
            k += 1
    for i in range(n):
        base = num_pairs + 4 * i
        A[base, i] = 1.0
        A[base + 1, i] = 1.0
        A[base + 2, i] = 1.0
        A[base + 3, i] = 1.0
    return A, pair_idx

A_LP, PAIR_IDX = build_lp_structure(N)
NUM_PAIRS = len(PAIR_IDX)

def solve_lp(centers):
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in PAIR_IDX:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if res.success:
        if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            duals = res.ineqlin.marginals
        elif hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            duals = res.marginals.ineqlin
        else:
            duals = np.zeros_like(b)
        return res.x, np.sum(res.x), duals
    return np.full(n, 1e-6), 0.0, np.zeros_like(b)

def compute_grad(centers, duals):
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    idx = 0
    for i, j in PAIR_IDX:
        mu = duals[idx]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = NUM_PAIRS
    for i in range(n):
        mu_L = duals[bound_start + 4*i]
        mu_R = duals[bound_start + 4*i + 1]
        mu_B = duals[bound_start + 4*i + 2]
        mu_T = duals[bound_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return grad

def lp_obj_grad(x_flat):
    c = x_flat.reshape(N, 2)
    _, s, duals = solve_lp(c)
    g = compute_grad(c, duals)
    return -s, -g.flatten()

def force_init(rng):
    c = rng.uniform(0.2, 0.8, (N, 2))
    for _ in range(1500):
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(c[i,0]-c[j,0], c[i,1]-c[j,1])
                if d < 0.25 and d > 1e-5:
                    f = (0.25 - d) / (d + 1e-5)
                    dx = c[i,0]-c[j,0]
                    dy = c[i,1]-c[j,1]
                    forces[i,0] += dx/d * f
                    forces[i,1] += dy/d * f
                    forces[j,0] -= dx/d * f
                    forces[j,1] -= dy/d * f
        forces[:,0] += np.where(c[:,0]<0.2, (c[:,0]-0.2), 0) + np.where(c[:,0]>0.8, (c[:,0]-0.8), 0)
        forces[:,1] += np.where(c[:,1]<0.2, (c[:,1]-0.2), 0) + np.where(c[:,1]>0.8, (c[:,1]-0.8), 0)
        c += forces * 0.05
        c = np.clip(c, 0.05, 0.95)
    return c

def hex_init(rng, pat):
    c = []
    r0 = 0.095
    y = r0
    for r_idx, cnt in enumerate(pat):
        shift = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
    c = np.array(c[:N])
    c = np.clip(c, 0.05, 0.95)
    return c

def slsqp_constraints(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    dx = c[TRIU_IND[0], 0] - c[TRIU_IND[1], 0]
    dy = c[TRIU_IND[0], 1] - c[TRIU_IND[1], 1]
    dr = r[TRIU_IND[0]] + r[TRIU_IND[1]]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def slsqp_objective(v):
    return -np.sum(v[2*N:])

def repair(centers, radii):
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5]
    ]
    
    starts = []
    for pat in patterns:
        starts.append(hex_init(rng, pat))
    for _ in range(10):
        starts.append(force_init(rng))
    for _ in range(5):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    lbfgs_bounds = [(0.01, 0.99)] * (2 * N)
    
    for c0 in starts:
        try:
            res = minimize(lp_obj_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=lbfgs_bounds, options={'maxiter': 4000, 'ftol': 1e-15})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_c = res.x.reshape(N, 2).copy()
                best_r, _, _ = solve_lp(best_c)
        except Exception:
            pass
            
    if best_c is not None:
        for step in range(25):
            noise = 0.004 * (0.85 ** step)
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            try:
                res = minimize(lp_obj_grad, c_pert.flatten(), method='L-BFGS-B', jac=True,
                               bounds=lbfgs_bounds, options={'maxiter': 2000, 'ftol': 1e-15})
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_c = res.x.reshape(N, 2).copy()
                    best_r, _, _ = solve_lp(best_c)
            except Exception:
                pass
                
        v0 = np.concatenate([best_c.flatten(), best_r])
        slsqp_bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        try:
            res = minimize(slsqp_objective, v0, method='SLSQP', bounds=slsqp_bounds,
                           constraints={'type': 'ineq', 'fun': slsqp_constraints},
                           options={'maxiter': 8000, 'ftol': 1e-14})
            if np.min(slsqp_constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
        best_r, s_lp, _ = solve_lp(best_c)
        if s_lp > best_sum:
            best_sum = s_lp
            best_r = best_r
            
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
