# sol_000222 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000123 (state 101aee21) state=42a6b81c sum of radii=2.606517 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute LP constraint matrix structure
# Constraints: r_i + r_j <= d_ij, r_i <= bound
# A_ub shape: (num_pairs + 4*N, N)
def setup_lp_A():
    num_pairs = N * (N - 1) // 2
    num_bound = 4 * N
    A = np.zeros((num_pairs + num_bound, N))
    pairs = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            pairs.append((i, j))
            idx += 1
    for i in range(N):
        for _ in range(4):
            A[idx, i] = 1.0
            idx += 1
    return A, pairs

LP_A, LP_PAIRS = setup_lp_A()
NUM_PAIRS = len(LP_PAIRS)

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes gradient via duals."""
    n = centers.shape[0]
    
    # Upper bounds for radii based on distance to boundaries
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    # Distances between centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # b_ub: distances for pairs, then boundary distances
    b_ub = np.zeros(LP_A.shape[0])
    idx = 0
    for i, j in LP_PAIRS:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        x, y = centers[i]
        b_ub[idx] = x; idx += 1
        b_ub[idx] = 1.0 - x; idx += 1
        b_ub[idx] = y; idx += 1
        b_ub[idx] = 1.0 - y; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0.0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=LP_A, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return np.zeros(n), -1.0, np.zeros_like(centers)
        
    radii = res.x
    duals = res.ineqlin.marginals
    
    # Gradient computation using LP duals
    grad = np.zeros_like(centers)
    idx = 0
    for i, j in LP_PAIRS:
        lam = duals[idx]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    # Boundary gradients
    b_start = NUM_PAIRS
    for i in range(n):
        mu_L = duals[b_start + 4*i]     # x >= r
        mu_R = duals[b_start + 4*i + 1] # 1-x >= r
        mu_B = duals[b_start + 4*i + 2] # y >= r
        mu_T = duals[b_start + 4*i + 3] # 1-y >= r
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, np.sum(radii), grad

def compute_constraints_sq(v):
    """Squared constraints for SLSQP to improve numerical stability"""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    idx = np.triu_indices(N, 1)
    dx = c[idx[0], 0] - c[idx[1], 0]
    dy = c[idx[0], 1] - c[idx[1], 1]
    dr = r[idx[0]] + r[idx[1]]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def obj_joint(v):
    return -np.sum(v[2*N:])

def generate_starts(rng):
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
    ]
    for pat in patterns:
        for r_est in [0.095, 0.10, 0.105, 0.11]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.02, 0.98)
            starts.append(c)
            
    # Random starts
    for _ in range(12):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Corner starts (exploits boundary space efficiently)
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = corners
        starts.append(c)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints_sq}
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient Ascent on Centers using LP Radii & Duals
    for c0 in starts:
        c = c0.copy()
        step = 0.005
        for k in range(2500):
            r_lp, s_lp, grad = solve_lp_and_grad(c)
            if s_lp > best_sum:
                best_sum = s_lp
                best_c = c.copy()
                best_r = r_lp.copy()
                
            gn = np.linalg.norm(grad)
            if gn < 1e-10:
                break
                
            c += step * (grad / gn)
            c = np.clip(c, 0.01, 0.99)
            
            if k % 120 == 0 and k > 0:
                step *= 0.96
            if k % 400 == 0 and k > 0:
                c += rng.normal(0, 0.0025, c.shape)
                c = np.clip(c, 0.02, 0.98)
                
    # Phase 2: SLSQP Joint Refinement (Centers + Radii)
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_opt,
                          constraints=cons, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if np.min(compute_constraints_sq(res.x)) >= -1e-7:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
    # Phase 3: Perturbation & SLSQP to escape local minima
    if best_c is not None:
        for i in range(60):
            noise = 0.007 * (0.88 ** (i // 4))
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            # Occasional coordinate swap to break symmetry traps
            if i % 5 == 0:
                swap_idx = rng.choice(N, 2, replace=False)
                c_pert[swap_idx] = c_pert[swap_idx[::-1]]
                
            r_pert, _, _ = solve_lp_and_grad(c_pert)
            v0 = np.concatenate([c_pert.flatten(), r_pert * 0.97])
            
            try:
                res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_opt,
                              constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14})
                if np.min(compute_constraints_sq(res.x)) >= -1e-7:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
            except Exception:
                continue
                
        # Final LP Polish to guarantee optimal radii for the converged geometry
        r_lp, s_lp, _ = solve_lp_and_grad(best_c)
        if s_lp > best_sum:
            best_r = r_lp
            best_sum = s_lp

    # Fallback if optimization unexpectedly fails
    if best_c is None:
        best_c = rng.uniform(0.1, 0.9, (N, 2))
        best_r = np.full(N, 0.05)
        best_sum = np.sum(best_r)

    # Strict Numerical Repair (Guarantees validator passes within 1e-12 tolerance)
    radii = best_r.copy()
    centers = best_c.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
