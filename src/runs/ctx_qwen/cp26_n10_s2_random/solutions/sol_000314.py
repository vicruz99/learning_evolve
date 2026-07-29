# sol_000314 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000286 (state d00da21c) state=9a7add96 sum of radii=2.500247 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure globally
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

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii, duals, sum."""
    n = centers.shape[0]
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
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
                      bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(N), np.zeros_like(b), 0.0
            
        radii = res.x
        sum_r = np.sum(radii)
        
        duals = np.zeros_like(b)
        try:
            duals = res.marginals.ineqlin
        except AttributeError:
            try:
                duals = res.ineqlin.marginals
            except AttributeError:
                pass
        return radii, duals, sum_r
    except Exception:
        return np.zeros(N), np.zeros_like(b), 0.0

def obj_grad(v):
    """Objective and exact gradient for L-BFGS-B."""
    c = v.reshape(N, 2)
    radii, duals, s = solve_lp(c)
    if s < 1e-9:
        return 0.0, np.zeros_like(v)
        
    obj = -s
    grad = np.zeros_like(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    # L-BFGS-B minimizes, so return negative gradient of sum_radii
    return obj, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], 
        [4, 5, 6, 5, 6], [5, 4, 6, 6, 5], [6, 5, 5, 6, 4],
        [5, 5, 4, 6, 6], [4, 6, 5, 5, 6], [6, 4, 5, 6, 5],
        [7, 6, 6, 7], [6, 7, 7, 6]
    ]
    for pat in patterns:
        for r_est in [0.090, 0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            while len(c) < N:
                c.append(rng.uniform(0.2, 0.8, 2))
            starts.append(np.array(c[:N]))
            
    # 2. Force-directed spreads
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        push = (0.22 - d) * 0.04
                        forces[i] += d_vec / d * push
                        forces[j] -= d_vec / d * push
            c += forces
            c = np.clip(c, 0.1, 0.9)
        starts.append(c)
        
    # 3. Corner/Edge biased starts
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = corners
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # 4. Pure random
    for _ in range(10):
        starts.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return starts

def try_boundary_push(c, rng):
    """Heuristic: try pushing each circle to boundaries/corners and optimize."""
    best_c = c.copy()
    _, _, best_s = solve_lp(c)
    
    directions = [
        [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
        [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]
    ]
    
    for _ in range(3):
        idx = rng.integers(N)
        for d in directions:
            c_try = c.copy()
            step = 0.005
            c_try[idx, 0] = np.clip(c[idx, 0] + d[0] * step, 0.01, 0.99)
            c_try[idx, 1] = np.clip(c[idx, 1] + d[1] * step, 0.01, 0.99)
            
            try:
                res = minimize(obj_grad, c_try.flatten(), method='L-BFGS-B', jac=True,
                              bounds=[(0.01, 0.99)] * (2 * N),
                              options={'maxiter': 800, 'ftol': 1e-12})
                s_val = -res.fun
                if s_val > best_s:
                    best_s = s_val
                    best_c = res.x.reshape(N, 2).copy()
            except Exception:
                continue
                
    return best_c, best_s

def repair(centers, radii):
    """Minimal repair to guarantee validation passes."""
    radii = radii.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) * 0.5 + 1e-10
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
    bounds_lbfgs = [(0.005, 0.995)] * (2 * N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B
    for c0 in starts:
        try:
            res = minimize(obj_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, 
                           options={'maxiter': 2500, 'ftol': 1e-13, 'gtol': 1e-12})
            s_val = -res.fun
            if s_val > best_sum:
                best_sum = s_val
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
    best_r, _, best_sum = solve_lp(best_c)
    
    # Phase 2: Simulated Annealing Perturbation
    T = 0.008
    for step in range(60):
        noise_scale = 0.006 * (1.0 + 0.5 * np.exp(-step / 15.0))
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, 0.01, 0.99)
        
        try:
            res = minimize(obj_grad, c_pert.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, 
                           options={'maxiter': 1200, 'ftol': 1e-13})
            c_pert = res.x.reshape(N, 2)
            r_pert, _, s_pert = solve_lp(c_pert)
            
            delta = s_pert - best_sum
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-6)):
                best_sum = s_pert
                best_c = c_pert.copy()
                best_r = r_pert.copy()
                if delta > 0:
                    T = min(T * 1.05, 0.02)
                else:
                    T *= 0.95
        except Exception:
            continue
            
    # Phase 3: Boundary Push Heuristic
    best_c, best_sum = try_boundary_push(best_c, rng)
    best_r, _, best_sum = solve_lp(best_c)
    
    # Phase 4: Final LP Verification & Repair
    lp_r, _, lp_sum = solve_lp(best_c)
    if lp_sum > best_sum:
        best_r = lp_r
        best_sum = lp_sum
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
