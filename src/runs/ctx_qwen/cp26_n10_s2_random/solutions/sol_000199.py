# sol_000199 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000169 (state 623e904f) state=e5589200 sum of radii=2.295231 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def setup_lp():
    """Precompute the sparse structure of the LP constraint matrix."""
    num_pairs = N * (N - 1) // 2
    A = np.zeros((num_pairs + 4 * N, N))
    p_idx = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[k, i] = 1.0
            A[k, j] = 1.0
            p_idx.append((i, j))
            k += 1
    for i in range(N):
        A[k, i] = 1.0; k += 1
        A[k, i] = 1.0; k += 1
        A[k, i] = 1.0; k += 1
        A[k, i] = 1.0; k += 1
    return A, p_idx

A_LP, PAIR_IDX = setup_lp()

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
    b = np.zeros(A_LP.shape[0])
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
            return np.full(N, 1e-9), 0.0, np.zeros_like(centers)
    except Exception:
        return np.full(N, 1e-9), 0.0, np.zeros_like(centers)
        
    radii = res.x
    duals = np.zeros(A_LP.shape[0])
    try:
        duals = np.maximum(np.asarray(res.ineqlin.marginals), 0.0)
    except Exception:
        pass
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4 * i] - duals[bound_start + 4 * i + 1]
        grad[i, 1] += duals[bound_start + 4 * i + 2] - duals[bound_start + 4 * i + 3]
        
    return radii, np.sum(radii), grad

def optimize_centers(c0):
    """Optimizes centers using L-BFGS-B with LP dual gradient."""
    bounds = [(0.005, 0.995)] * (2 * N)
    
    def obj_grad(v):
        c_mat = v.reshape(N, 2)
        _, val, g = solve_lp_and_grad(c_mat)
        return -val, -g.flatten()
        
    try:
        res = minimize(obj_grad, c0.flatten(), method='L-BFGS-B', bounds=bounds, 
                       jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [5, 6, 6, 4, 5], [6, 6, 4, 5, 5],
        [5, 5, 5, 6, 5], [5, 4, 6, 6, 5], [4, 5, 6, 5, 6]
    ]
    
    for pat in patterns:
        for r0 in [0.090, 0.096, 0.102, 0.108]:
            for shift_odd in [True, False]:
                c = []
                y = r0
                for r_idx, cnt in enumerate(pat):
                    start_x = 2 * r0 if (shift_odd and r_idx % 2 == 1) or (not shift_odd and r_idx % 2 == 0) else r0
                    x = start_x
                    for _ in range(cnt):
                        if len(c) < N:
                            c.append([x, y])
                        x += 2.0 * r0
                    y += r0 * np.sqrt(3.0)
                c = np.array(c[:N])
                c += rng.normal(0, 0.0025, c.shape)
                c = np.clip(c, 0.05, 0.95)
                starts.append(c)
                
    # Random dense starts
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to ensure strict validity."""
    radii = radii.copy()
    for _ in range(80):
        changed = False
        # Boundary clamp
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        # Overlap resolution
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
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: L-BFGS-B optimization from diverse starts
    for c_init in starts:
        c_opt, s_opt = optimize_centers(c_init)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: Perturbation & Re-optimize to escape local minima
    if best_c is not None:
        for step in range(25):
            noise_scale = 0.008 * (0.85 ** step)
            c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            c_opt2, s_opt2 = optimize_centers(c_pert)
            if s_opt2 > best_sum:
                best_sum = s_opt2
                best_c = c_opt2.copy()
                
    # Final LP solve for exact radii matching best centers
    if best_c is not None:
        best_r, final_s, _ = solve_lp_and_grad(best_c)
        if final_s > best_sum:
            best_sum = final_s
            
    # Strict numerical repair
    centers = best_c.copy()
    radii = repair(centers, best_r.copy())
    
    return centers, radii, float(np.sum(radii))
