# sol_000356 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000317 (state f476b79f) state=f3308817 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure (constant across runs)
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_IDX.append((i, j))
        idx += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii and computes exact subgradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-24)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
    return radii, s_sum, grad

def obj_centers(x):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    centers = x.reshape(N, 2)
    _, s, g = solve_lp_and_grad(centers)
    return -s, -g.flatten()

def constraints_joint(v):
    """Computes boundary and non-overlap constraints for joint SLSQP optimization."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def force_init(rng):
    """Generates a well-spaced configuration via repulsive forces."""
    c = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(800):
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i+1, N):
                dv = c[i] - c[j]
                d = np.linalg.norm(dv)
                if d < 0.2 and d > 1e-4:
                    push = (0.2 - d) * 0.05 / (d + 1e-4)
                    f[i] += dv / d * push
                    f[j] -= dv / d * push
        c += f
        c = np.clip(c, 0.05, 0.95)
    return c

def subset_obj(x_sub, centers_fixed, subset_idx):
    """Objective for subset optimization."""
    centers = centers_fixed.copy()
    centers[subset_idx] = x_sub.reshape(len(subset_idx), 2)
    _, s, g = solve_lp_and_grad(centers)
    return -s, -g[subset_idx].flatten()

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
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

def run_packing():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5]]
    for pat in pats:
        for r0 in [0.092, 0.098, 0.105, 0.110]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    for _ in range(10):
        starts.append(force_init(rng))
    for _ in range(10):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))

    bounds_c = [(0.001, 0.999)] * (2 * N)
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c_init in starts:
        try:
            res1 = minimize(obj_centers, c_init.flatten(), jac=True, method='L-BFGS-B', 
                            bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-14})
            c1 = res1.x.reshape(N, 2)
            r1, s1, _ = solve_lp_and_grad(c1)
            if s1 > best_s:
                best_s = s1
                best_c = c1.copy()
                best_r = r1.copy()
        except Exception:
            pass

    # Phase 2: Joint SLSQP Polish on best
    if best_c is not None:
        try:
            ub = np.minimum(np.minimum(best_c[:, 0], 1.0 - best_c[:, 0]), 
                            np.minimum(best_c[:, 1], 1.0 - best_c[:, 1]))
            dists = np.linalg.norm(best_c[:, None, :] - best_c[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            rp = 0.5 * np.min(dists, axis=1)
            r0_init = np.minimum(ub, rp) * 0.85
            
            v0 = np.concatenate([best_c.flatten(), r0_init])
            res2 = minimize(objective_joint, v0, method='SLSQP', bounds=bounds_joint,
                            constraints={'type': 'ineq', 'fun': constraints_joint},
                            options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if np.min(constraints_joint(res2.x)) >= -1e-7:
                s2 = np.sum(res2.x[2*N:])
                if s2 > best_s:
                    best_s = s2
                    best_c = res2.x[:2*N].reshape(N, 2).copy()
                    best_r = res2.x[2*N:].copy()
        except Exception:
            pass

    # Phase 3: Subset Optimization to break symmetry
    for _ in range(150):
        sub_idx = rng.choice(N, size=rng.integers(5, 11), replace=False)
        other_idx = np.setdiff1d(np.arange(N), sub_idx)
        centers_fixed = best_c.copy()
        
        try:
            res_sub = minimize(subset_obj, best_c[sub_idx].flatten(), jac=True, args=(centers_fixed, sub_idx),
                               method='L-BFGS-B', bounds=bounds_c[2*sub_idx[0]:2*sub_idx[-1]+2],
                               options={'maxiter': 1000, 'ftol': 1e-13})
            centers_fixed[sub_idx] = res_sub.x.reshape(len(sub_idx), 2)
            r_sub, s_sub, _ = solve_lp_and_grad(centers_fixed)
            if s_sub > best_s:
                best_s = s_sub
                best_c = centers_fixed.copy()
                best_r = r_sub.copy()
        except Exception:
            pass

    # Phase 4: Basin Hopping / Simulated Annealing with adaptive step
    c_bh = best_c.copy()
    s_bh = best_s
    T = 0.006
    for step in range(800):
        noise_scale = 0.004 * (0.98 ** (step // 20))
        c_try = c_bh + rng.normal(0, noise_scale, c_bh.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        if s_try > s_bh or (s_bh > 0 and np.exp((s_try - s_bh) / max(T, 1e-9)) > rng.random()):
            c_bh, s_bh = c_try, s_try
            if s_bh > best_s:
                best_s = s_bh
                best_c = c_bh.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.995
        
    # Re-optimize SA result with L-BFGS-B
    try:
        res_sa = minimize(obj_centers, best_c.flatten(), jac=True, method='L-BFGS-B', 
                          bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-14})
        c_sa = res_sa.x.reshape(N, 2)
        r_sa, s_sa, _ = solve_lp_and_grad(c_sa)
        if s_sa > best_s:
            best_s = s_sa
            best_c = c_sa.copy()
            best_r = r_sa.copy()
    except Exception:
        pass

    # Phase 5: Swap-based exploration
    for _ in range(80):
        c_swap = best_c.copy()
        i, j = rng.choice(N, 2, replace=False)
        c_swap[i], c_swap[j] = c_swap[j], c_swap[i]
        
        try:
            res_sw = minimize(obj_centers, c_swap.flatten(), jac=True, method='L-BFGS-B',
                              bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-13})
            c_sw2 = res_sw.x.reshape(N, 2)
            r_sw2, s_sw2, _ = solve_lp_and_grad(c_sw2)
            if s_sw2 > best_s:
                best_s = s_sw2
                best_c = c_sw2.copy()
                best_r = r_sw2.copy()
        except Exception:
            pass

    # Phase 6: Final High-Precision Joint Polish
    try:
        v0_final = np.concatenate([best_c.flatten(), best_r])
        res_final = minimize(objective_joint, v0_final, method='SLSQP', bounds=bounds_joint,
                             constraints={'type': 'ineq', 'fun': constraints_joint},
                             options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
        if np.min(constraints_joint(res_final.x)) >= -1e-8:
            s_final = np.sum(res_final.x[2*N:])
            if s_final > best_s:
                best_c = res_final.x[:2*N].reshape(N, 2)
                best_r = res_final.x[2*N:]
                best_s = s_final
    except Exception:
        pass
        
    # Phase 7: Strict numerical repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    # Fallback update if repair slightly decreased sum but LP had better
    if final_sum < best_s - 1e-6:
        radii, final_sum, _ = solve_lp_and_grad(best_c)
        
    return best_c, radii, final_sum
