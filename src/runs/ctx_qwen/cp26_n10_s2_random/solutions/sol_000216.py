# sol_000216 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000168 (state 79899e79) state=fcc6329b sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute LP constraint structure (constant across runs)
num_pairs = N * (N - 1) // 2
A_ub = np.zeros((num_pairs + 4 * N, N))
pair_idx = []
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        pair_idx.append((i, j))
        k += 1
for i in range(N):
    base = num_pairs + 4 * i
    A_ub[base, i] = 1.0      # r_i <= x_i
    A_ub[base+1, i] = 1.0    # r_i <= 1-x_i
    A_ub[base+2, i] = 1.0    # r_i <= y_i
    A_ub[base+3, i] = 1.0    # r_i <= 1-y_i

def solve_lp(centers):
    """Solves LP to find maximum sum of radii for fixed centers and returns duals."""
    n = centers.shape[0]
    dx = centers[:, 0, None] - centers[None, :, 0]
    dy = centers[:, 1, None] - centers[None, :, 1]
    dists = np.hypot(dx, dy)
    
    b = np.zeros(A_ub.shape[0])
    idx = 0
    for i, j in pair_idx:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b, bounds=(0, None), method='highs')
    if res.success:
        return res.x, -res.fun, res.ineqlin.marginals
    return np.zeros(n), 0.0, np.zeros_like(b)

def obj_grad(v):
    """Objective and gradient for center optimization. Minimizes negative sum of radii."""
    c = v.reshape(N, 2)
    radii, val, duals = solve_lp(c)
    grad = np.zeros_like(c)
    
    idx = 0
    for i, j in pair_idx:
        mu = duals[idx]
        if mu > 1e-9:
            d = np.hypot(c[i,0]-c[j,0], c[i,1]-c[j,1])
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = len(pair_idx)
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
        
    return -val, -grad.flatten()

def get_start_configs(rng):
    """Generates diverse initial configurations."""
    configs = []
    # Hexagonal patterns with varying densities
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], [6,6,5,5,4], [5,4,6,6,5]]
    for pat in pats:
        for r0 in [0.08, 0.09, 0.10, 0.11, 0.115]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N: c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            c = np.array(c[:N]) + rng.normal(0, 0.003, (N,2))
            c = np.clip(c, 0.05, 0.95)
            configs.append(c)
            
    # Force-directed layouts to explore non-lattice arrangements
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i]-c[j])
                    if 1e-4 < d < 0.22:
                        f = (0.22 - d)/d * 0.0015
                        diff = c[i]-c[j]
                        c[i] += diff * f
                        c[j] -= diff * f
            c = np.clip(c, 0.1, 0.9)
        configs.append(c)
        
    # Pure random starts
    for _ in range(20):
        configs.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_v = None
    best_sum = -1.0
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    
    starts = get_start_configs(rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c0 in starts:
        v0 = c0.flatten()
        try:
            res = minimize(obj_grad, v0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Basin-hopping / Perturbation refinement
    if best_v is not None:
        c_curr = best_v.reshape(N, 2)
        s_curr = best_sum
        for step in range(40):
            noise = 0.006 * (0.82 ** step)
            c_pert = c_curr + rng.normal(0, noise, c_curr.shape)
            c_pert = np.clip(c_pert, 1e-4, 1.0 - 1e-4)
            try:
                res = minimize(obj_grad, c_pert.flatten(), method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
                if -res.fun > s_curr:
                    s_curr = -res.fun
                    c_curr = res.x.reshape(N, 2).copy()
            except Exception:
                pass
        best_v = c_curr.flatten()
        
    centers = best_v.reshape(N, 2)
    radii, final_sum, _ = solve_lp(centers)
    
    # Phase 3: Joint SLSQP polish (centers + radii) to resolve LP flat regions
    def slsqp_obj(v):
        return -np.sum(v[2*N:])
    def slsqp_cons(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
        idx = np.triu_indices(N, 1)
        d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
        con.append(d - (r[idx[0]] + r[idx[1]]))
        return np.concatenate(con)
        
    v0_sl = np.concatenate([centers.flatten(), radii])
    b_sl = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    try:
        res_sl = minimize(slsqp_obj, v0_sl, method='SLSQP', bounds=b_sl,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res_sl.x)) >= -1e-7:
            c_sl = res_sl.x[:2*N].reshape(N, 2)
            r_sl = res_sl.x[2*N:]
            if np.sum(r_sl) > final_sum:
                centers = c_sl
                radii = r_sl
                final_sum = np.sum(radii)
    except Exception:
        pass
        
    # Phase 4: Strict deterministic repair for validation compliance
    for _ in range(100):
        changed = False
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed: break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
