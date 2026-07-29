# sol_000215 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000168 (state 79899e79) state=f5dfd5b6 sum of radii=2.268101 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

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

def solve_lp_radii(centers):
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
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if res.success:
            duals = res.marginals.ineqlin if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin') else res.ineqlin.marginals
            return res.x, np.sum(res.x), duals
    except Exception:
        pass
    return np.ones(n) * 1e-6, 26e-6, np.zeros_like(b)

def compute_gradient(centers, duals):
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    idx = 0
    for i, j in PAIR_IDX:
        mu = duals[idx]
        if mu > 1e-10:
            d = dists[i, j]
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(n):
        mu_L = duals[bound_start + 4*i]
        mu_R = duals[bound_start + 4*i + 1]
        mu_B = duals[bound_start + 4*i + 2]
        mu_T = duals[bound_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return grad

def lbfgs_obj_grad(x):
    centers = x.reshape(N, 2)
    radii, s, duals = solve_lp_radii(centers)
    grad = compute_gradient(centers, duals)
    return -s, -grad.flatten()

def generate_inits(rng):
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5]
    ]
    for pat in patterns:
        for r0 in [0.085, 0.092, 0.098, 0.105, 0.112]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.03, 0.97)
            inits.append(c)
            
    for _ in range(30):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(300):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        f = (0.22 - d) / d * 0.015
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces
            c = np.clip(c, 0.03, 0.97)
        inits.append(c)
        
    return inits

def repair(centers, radii):
    radii = radii.copy()
    for _ in range(200):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    bounds_c = [(0.0, 1.0)] * (2 * N)
    inits = generate_inits(rng)
    
    # Phase 1: L-BFGS-B from multiple starts
    for i, c_init in enumerate(inits):
        x0 = c_init.flatten()
        try:
            res = minimize(lbfgs_obj_grad, x0, method='L-BFGS-B', bounds=bounds_c,
                           jac=True, options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-10})
            s_curr = -res.fun
            if np.isfinite(s_curr) and s_curr > best_sum:
                best_sum = s_curr
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is not None:
        # Phase 2: Perturbation & Refinement to escape local minima
        for step in range(25):
            noise = 0.006 * (0.82 ** step)
            c_per = best_c + rng.normal(0, noise, best_c.shape)
            c_per = np.clip(c_per, 0.02, 0.98)
            try:
                res = minimize(lbfgs_obj_grad, c_per.flatten(), method='L-BFGS-B', bounds=bounds_c,
                               jac=True, options={'maxiter': 2500, 'ftol': 1e-13})
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_c = res.x.reshape(N, 2).copy()
            except Exception:
                pass
                
        # Phase 3: Final LP polish to extract exact maximal radii
        r_lp, s_lp, _ = solve_lp_radii(best_c)
        if s_lp > best_sum:
            best_sum = s_lp
        best_r = r_lp
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
