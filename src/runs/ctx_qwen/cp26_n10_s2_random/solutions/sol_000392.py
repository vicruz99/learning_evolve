# sol_000392 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000360 (state f30435ab) state=87c8d8d3 sum of radii=2.609174 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

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
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
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
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                      bounds=[(0.0, u) for u in ub], method='highs')
    except Exception:
        return np.full(N, 0.08), 0.0, np.zeros_like(centers)
        
    if not res.success:
        return np.full(N, 0.08), 0.0, np.zeros_like(centers)
        
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
        if mu > 1e-10:
            d = dists[i, j]
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def obj_lp_centers(x):
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

def generate_diverse_starts(rng):
    """Generates a wide variety of initial configurations."""
    starts = []
    
    # 1. Hexagonal lattices with various row distributions
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5]]
    for pat in pats:
        for r0 in [0.085, 0.095, 0.105, 0.115]:
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
            
    # 2. Corner & Edge biased layouts
    for _ in range(10):
        c = np.zeros((N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        # Edges
        c[4:8] = [[rng.uniform(0.2, 0.8), 0.08], [rng.uniform(0.2, 0.8), 0.92],
                  [0.08, rng.uniform(0.2, 0.8)], [0.92, rng.uniform(0.2, 0.8)]]
        # Center fill
        c[8:] = rng.uniform(0.3, 0.7, (18, 2))
        c += rng.normal(0, 0.005, c.shape)
        starts.append(np.clip(c, 0.02, 0.98))

    # 3. Force-directed relaxation from random
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        push = (0.25 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f * 0.5
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)

    # 4. Uniform random
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
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
    best_s = -1.0
    
    bounds_c = [(0.002, 0.998)] * (2 * N)
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    starts = generate_diverse_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B
    for c_init in starts:
        try:
            res = minimize(obj_lp_centers, c_init.flatten(), jac=True, method='L-BFGS-B', 
                           bounds=bounds_c, options={'maxiter': 6000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass

    # Phase 2: Simulated Annealing with Cluster Breaks
    if best_c is not None:
        c_curr = best_c.copy()
        s_curr = best_s
        T = 0.015
        
        for step in range(2500):
            # Determine perturbation scale based on temperature
            scale = T * 0.4
            
            # Every 150 steps, do a "cluster break" by moving a random subset
            if step % 150 == 0 and step > 0:
                k = rng.integers(5, 12)
                idx = rng.choice(N, k, replace=False)
                c_try = c_curr.copy()
                c_try[idx] = rng.uniform(0.1, 0.9, (k, 2))
            else:
                c_try = c_curr + rng.normal(0, scale, c_curr.shape)
                
            c_try = np.clip(c_try, 0.02, 0.98)
            _, s_try, _ = solve_lp_and_grad(c_try)
            
            delta = s_try - s_curr
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                c_curr = c_try
                s_curr = s_try
                if s_curr > best_s:
                    best_s = s_curr
                    best_c = c_curr.copy()
                    best_r, _, _ = solve_lp_and_grad(best_c)
            
            T *= 0.9975
            
            # Periodic local L-BFGS-B to settle after SA moves
            if step % 100 == 0 and step > 0:
                try:
                    res_loc = minimize(obj_lp_centers, c_curr.flatten(), jac=True, method='L-BFGS-B', 
                                       bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-13})
                    c_loc = res_loc.x.reshape(N, 2)
                    r_loc, s_loc, _ = solve_lp_and_grad(c_loc)
                    if s_loc > s_curr:
                        c_curr = c_loc
                        s_curr = s_loc
                        if s_loc > best_s:
                            best_s = s_loc
                            best_c = c_loc.copy()
                            best_r = r_loc.copy()
                except Exception:
                    pass

    # Phase 3: Pairwise Swapping & Local Optimization
    if best_c is not None:
        for _ in range(200):
            i, j = rng.choice(N, 2, replace=False)
            c_sw = best_c.copy()
            c_sw[[i, j]] = c_sw[[j, i]]
            try:
                res_sw = minimize(obj_lp_centers, c_sw.flatten(), jac=True, method='L-BFGS-B',
                                  bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-13})
                c_sw_opt = res_sw.x.reshape(N, 2)
                r_sw, s_sw, _ = solve_lp_and_grad(c_sw_opt)
                if s_sw > best_s:
                    best_s = s_sw
                    best_c = c_sw_opt.copy()
                    best_r = r_sw.copy()
            except Exception:
                pass

    # Phase 4: Joint SLSQP High-Precision Polish
    if best_c is not None:
        # Initialize radii for joint opt
        ub = np.minimum(np.minimum(best_c[:, 0], 1.0 - best_c[:, 0]), 
                        np.minimum(best_c[:, 1], 1.0 - best_c[:, 1]))
        dists = np.linalg.norm(best_c[:, None, :] - best_c[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(ub, rp) * 0.90
        
        v0 = np.concatenate([best_c.flatten(), r_init])
        try:
            res_joint = minimize(objective_joint, v0, method='SLSQP', bounds=bounds_joint,
                                 constraints={'type': 'ineq', 'fun': constraints_joint},
                                 options={'maxiter': 15000, 'ftol': 1e-15, 'disp': False})
            if np.min(constraints_joint(res_joint.x)) >= -1e-7:
                s_joint = np.sum(res_joint.x[2*N:])
                if s_joint > best_s:
                    best_s = s_joint
                    best_c = res_joint.x[:2*N].reshape(N, 2).copy()
                    best_r = res_joint.x[2*N:].copy()
        except Exception:
            pass
            
        # Final LP sync to ensure radii match centers exactly
        lp_r, lp_s, _ = solve_lp_and_grad(best_c)
        if lp_s > best_s - 1e-9:
            best_r = lp_r
            best_s = lp_s

    # Phase 5: Strict Repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    return best_c, radii, final_sum
