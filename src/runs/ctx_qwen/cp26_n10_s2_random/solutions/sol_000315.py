# sol_000315 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000286 (state d00da21c) state=41300c0e sum of radii=2.621946 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure globally for speed
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
    """Solves LP for maximal radii given fixed centers and computes exact gradient via duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
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
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract duals safely across scipy versions
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = np.asarray(res.marginals.ineqlin)
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = np.asarray(res.ineqlin.marginals)
        
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

def obj_and_grad(x_flat):
    """Objective and gradient wrapper for optimizers."""
    centers = np.clip(x_flat.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(centers)
    return -val, -grad.flatten()

def generate_starts(rng):
    """Generates a diverse set of high-quality initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [5, 4, 6, 6, 5], [6, 5, 5, 6, 4],
        [5, 5, 4, 6, 6], [4, 6, 5, 5, 6], [6, 4, 5, 6, 5],
        [7, 6, 6, 7], [6, 7, 7, 6], [5, 7, 7, 7]
    ]
    
    for pat in patterns:
        for scale in [0.95, 1.0, 1.05, 1.10]:
            r_est = 0.102 / scale
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            # Normalize to [0.05, 0.95]
            c = (c - c.min(axis=0)) / (c.max(axis=0) - c.min(axis=0) + 1e-8)
            c = c * 0.85 + 0.075
            c += rng.normal(0, 0.003, c.shape)
            starts.append(np.clip(c, 0.02, 0.98))
            
    # Force-directed spreads
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(1000):
            forces = np.zeros_like(c)
            diffs = c[:, None, :] - c[None, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-8)
            np.fill_diagonal(dists, np.inf)
            mask = dists < 0.25
            rep = np.zeros_like(dists)
            rep[mask] = 0.02 / (dists[mask]**2)
            for d in range(2):
                forces[:, d] = np.sum(diffs[:, :, d] * rep, axis=1)
            # Boundary repulsion
            forces += np.where(c < 0.1, 0.05 * (0.1 - c), 0.0)
            forces -= np.where(c > 0.9, 0.05 * (c - 0.9), 0.0)
            c += forces * 0.005
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Random starts
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

def obj_joint(v):
    return -np.sum(v[2 * N:])

def cons_joint_sq(v):
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.0, 1.0)] * (2 * N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B with exact LP gradient
    for i, c0 in enumerate(starts):
        try:
            res = minimize(obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 5000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Swap & Perturbation Local Search to escape symmetry traps
    for step in range(60):
        c_trial = best_c.copy()
        idx = rng.choice(N, size=N, replace=True)
        # Small random jitter
        c_trial[idx] += rng.normal(0, 0.004 * (0.95**step), (N, 2))
        # Swap two random circles
        swap_a, swap_b = rng.choice(N, 2, replace=False)
        c_trial[swap_a], c_trial[swap_b] = c_trial[swap_b], c_trial[swap_a]
        c_trial = np.clip(c_trial, 0.01, 0.99)
        
        try:
            res = minimize(obj_and_grad, c_trial.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 2000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Joint SLSQP Polish on centers and radii
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    for _ in range(3):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                          constraints={'type': 'ineq', 'fun': cons_joint_sq},
                          options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if np.min(cons_joint_sq(res.x)) >= -1e-7:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Final LP verification to ensure radii are optimal for the final centers
    lp_r, lp_s, _ = solve_lp_and_grad(best_c)
    if lp_s > best_sum:
        best_r = lp_r
        best_sum = lp_s
        
    # Strict numerical repair
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
