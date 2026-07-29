# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000124 (state e4120b9c) state=27fd9551 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp_radii(centers):
    """Solves LP to find maximum sum of radii for fixed centers."""
    c = np.clip(centers, 1e-9, 1.0 - 1e-9)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 0.0)
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    n_pairs = N * (N - 1) // 2
    A_ub = np.zeros((n_pairs, N))
    b_ub = np.zeros(n_pairs)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        return res.x, np.sum(res.x)
    return np.zeros(N), 0.0

def joint_objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def joint_constraints(v):
    """Computes boundary and non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = []
    
    # Boundary constraints
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist(i,j) - (r_i + r_j) >= 0
    idx_i, idx_j = np.triu_indices(N, 1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    dr = r[idx_i] + r[idx_j]
    cons.append(np.sqrt(dx**2 + dy**2) - dr)
    
    return np.concatenate(cons)

def lp_objective_func(x):
    """Objective for centers-only optimization: minimize negative LP sum of radii."""
    _, s = solve_lp_radii(x.reshape(N, 2))
    return -s

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    bounds_joint = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    bounds_c = [(0.0, 1.0)] * (2*N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # --- Phase 1: Generate Diverse Initial Configurations ---
    starts = []
    
    # 1. Hexagonal lattice patterns with varying row structures
    patterns = [[5,5,5,5,6], [5,6,5,6,4], [6,5,6,5,4], [4,6,6,6,4], [5,5,5,6,5]]
    for pat in patterns:
        c = []
        r_est = 0.095
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(c) < N:
                    c.append([x, y])
                x += 2.0 * r_est
            y += r_est * 1.7320508
        starts.append(np.array(c[:N]))
        
    # 2. Force-repulsion spread configurations
    for _ in range(5):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            dist = np.maximum(dist, 1e-4)
            f += np.sum(diff / (dist**2)[:, :, None], axis=1)
            c += 0.003 * f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # --- Phase 2: SLSQP Joint Optimization ---
    for c_init in starts:
        # Conservative feasible radii initialization
        rb = np.minimum(np.minimum(c_init[:,0], 1-c_init[:,0]), np.minimum(c_init[:,1], 1-c_init[:,1]))
        dists = np.linalg.norm(c_init[:,None,:] - c_init[None,:, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(rb, rp) * 0.8
        
        v0 = np.concatenate([c_init.flatten(), r_init])
        
        # Multiple slight perturbations per start
        for _ in range(2):
            v_curr = v0 + rng.normal(0, 0.002, v0.shape)
            v_curr = np.clip(v_curr, 0.01, 0.99)
            v_curr[2*N:] = np.clip(v_curr[2*N:], 0.01, 0.4)
            
            try:
                res = minimize(joint_objective, v_curr, method='SLSQP', bounds=bounds_joint,
                              constraints={'type': 'ineq', 'fun': joint_constraints},
                              options={'maxiter': 6000, 'ftol': 1e-12})
                if np.min(joint_constraints(res.x)) >= -1e-8:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
            except Exception:
                pass

    # --- Phase 3: LP Refinement & Centers-Only Powell Search ---
    if best_c is not None:
        # Exact LP radii for best centers so far
        r_lp, s_lp = solve_lp_radii(best_c)
        if s_lp > best_sum:
            best_sum = s_lp
            best_r = r_lp
            
        # Powell optimization on centers only (non-smooth but robust)
        x0 = best_c.flatten()
        try:
            res_p = minimize(lp_objective_func, x0, method='Powell', bounds=bounds_c,
                            options={'maxiter': 1200, 'ftol': 1e-12})
            c_powell = res_p.x.reshape(N, 2)
            r_powell, s_powell = solve_lp_radii(c_powell)
            if s_powell > best_sum:
                best_sum = s_powell
                best_c = c_powell
                best_r = r_powell
        except Exception:
            pass
            
        # Perturb & Powell to escape shallow basins
        for _ in range(3):
            c_pert = best_c + rng.normal(0, 0.004, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            try:
                res_p2 = minimize(lp_objective_func, c_pert.flatten(), method='Powell', bounds=bounds_c,
                                 options={'maxiter': 900, 'ftol': 1e-12})
                c2 = res_p2.x.reshape(N, 2)
                r2, s2 = solve_lp_radii(c2)
                if s2 > best_sum:
                    best_sum = s2
                    best_c = c2
                    best_r = r2
            except Exception:
                pass

    # --- Phase 4: Strict Numerical Repair ---
    centers = best_c.copy()
    radii = best_r.copy()
    
    for _ in range(60):
        changed = False
        
        # Resolve overlaps by symmetric shrinking
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Clamp to boundaries
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
                
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
