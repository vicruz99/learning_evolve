# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000111 (state 4b754d5d) state=101aee21 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_max_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 0.0)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    constraints = []
    rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints.append(row)
            rhs.append(dists[i, j])
            
    A_ub = np.array(constraints)
    b_ub = np.array(rhs)
    c_obj = -np.ones(n)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, u) for u in ub], method='highs')
    if res.success:
        return res.x
    return np.zeros(n)

def obj_func(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def compute_constraints(v):
    """Computes boundary and non-overlap constraints. Must be >= 0."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    idx = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
    con.append(d - (r[idx[0]] + r[idx[1]]))
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = []
    
    # 1. Hexagonal patterns with various row distributions
    patterns = [[5,5,5,5,6], [5,6,5,6,4], [6,5,6,5,4], [4,6,6,6,4], [6,6,5,5,4], [5,5,6,5,5]]
    for pat in patterns:
        c_hex = []
        r_est = 0.1
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(c_hex) < N:
                    c_hex.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        while len(c_hex) < N:
            c_hex.append(rng.uniform(0.1, 0.9, 2))
        inits.append(np.array(c_hex[:N]))
        
    # 2. Uniform random starts
    for _ in range(12):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # 3. Corner-focused starts (exploits boundary space efficiently)
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = corners
        inits.append(c)

    # Primary Optimization Phase
    for c0 in inits:
        r0 = compute_max_radii_lp(c0) * 0.95
        v0 = np.concatenate([c0.flatten(), r0])
        
        # Slight perturbation to avoid degenerate gradients
        v0_pert = v0 + rng.normal(0, 0.0005, v0.shape)
        v0_pert = np.clip(v0_pert, 0.01, 0.99)
        v0_pert[2*N:] = np.maximum(v0_pert[2*N:], 0.01)
        
        try:
            res = minimize(obj_func, v0_pert, method='SLSQP', bounds=bounds_opt, 
                          constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            if np.min(compute_constraints(res.x)) >= -1e-8:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass

    # Iterative Perturbation & Refinement Phase (Escapes local minima)
    if best_c is not None:
        for i in range(40):
            noise = 0.01 * (0.92 ** (i // 4))
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            # Swap two random circles to break symmetric traps
            swap_idx = rng.choice(N, 2, replace=False)
            c_pert[swap_idx] = c_pert[swap_idx[::-1]]
            
            r_pert = compute_max_radii_lp(c_pert) * 0.98
            v0 = np.concatenate([c_pert.flatten(), r_pert])
            
            try:
                res = minimize(obj_func, v0, method='SLSQP', bounds=bounds_opt, 
                              constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
                if np.min(compute_constraints(res.x)) >= -1e-8:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
            except Exception:
                continue
                
        # Final LP refinement on the best centers to guarantee optimal radii for that layout
        lp_r = compute_max_radii_lp(best_c)
        if np.sum(lp_r) > best_sum:
            best_r = lp_r
            best_sum = np.sum(lp_r)

    # Fallback if optimization unexpectedly fails
    if best_c is None:
        best_c = rng.uniform(0.1, 0.9, (N, 2))
        best_r = compute_max_radii_lp(best_c)
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
