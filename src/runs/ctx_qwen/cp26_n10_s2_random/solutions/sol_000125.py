# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000111 (state 4b754d5d) state=da577458 sum of radii=2.620354 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_radii_fast(centers):
    """Fast vectorized approximation of max radii for given centers."""
    c = centers.copy()
    rb = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    dists = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    r = np.minimum(rb, rp)
    return np.maximum(r, 0.0)

def obj_centers(v):
    """Objective for center-only optimization: maximize sum of radii."""
    c = v.reshape(N, 2)
    return -np.sum(compute_radii_fast(c))

def get_max_radii_lp(centers):
    """Exact LP solver for max sum of radii given fixed centers."""
    c = np.clip(centers, 1e-9, 1.0 - 1e-9)
    b = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                   np.minimum(c[:, 1], 1.0 - c[:, 1]))
    dists = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    c_obj = -np.ones(N)
    m = N*(N-1)//2 + N
    A_ub = np.zeros((m, N))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub[idx] = row
            b_ub[idx] = dists[i, j]
            idx += 1
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_ub[idx] = row
        b_ub[idx] = b[i]
        idx += 1
        
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*N, method='highs')
    if res.success:
        return res.x
    return np.zeros(N)

def joint_obj(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def joint_constraints(v):
    """Boundary and squared-overlap constraints for SLSQP (>= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    idx_i, idx_j = np.triu_indices(N, 1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    dist_sq = dx*dx + dy*dy
    r_sum_sq = (r[idx_i] + r[idx_j])**2
    con.append(dist_sq - r_sum_sq)
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_c = [(0.0, 1.0)] * (2*N)
    bounds_r = [(0.0, 0.5)] * N
    bounds_joint = bounds_c + bounds_r
    
    cons_joint = {'type': 'ineq', 'fun': joint_constraints}
    
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # Phase 1: Diverse starts for center optimization
    starts = []
    
    # Hex grids with various densities
    for density in [0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11]:
        pts = []
        y = density
        row = 0
        while len(pts) < N:
            shift = density if row % 2 == 1 else 0.0
            x = density + shift
            while x + density <= 1.0 and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * density
            y += density * np.sqrt(3)
            row += 1
        starts.append(np.array(pts[:N]))
        
    # Random starts
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Corner/Edge focused starts
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        starts.append(c)
        
    # Optimize centers
    for s in starts:
        try:
            # L-BFGS-B (fast, respects bounds)
            res = minimize(obj_centers, s.flatten(), method='L-BFGS-B', bounds=bounds_c,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            c_opt = res.x.reshape(N, 2)
            r_opt = compute_radii_fast(c_opt)
            s_val = np.sum(r_opt)
            if s_val > best_sum:
                best_sum = s_val
                best_c = c_opt.copy()
                best_r = r_opt.copy()
                
            # Powell (robust for non-smooth landscapes)
            res2 = minimize(obj_centers, s.flatten(), method='Powell', bounds=bounds_c,
                            options={'maxiter': 2000, 'ftol': 1e-12, 'xtol': 1e-12})
            c_opt2 = res2.x.reshape(N, 2)
            r_opt2 = compute_radii_fast(c_opt2)
            s_val2 = np.sum(r_opt2)
            if s_val2 > best_sum:
                best_sum = s_val2
                best_c = c_opt2.copy()
                best_r = r_opt2.copy()
        except Exception:
            pass
            
    # Phase 2: Joint SLSQP refinement
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        
        try:
            res = minimize(joint_obj, v0, method='SLSQP', bounds=bounds_joint,
                           constraints=cons_joint, options={'maxiter': 15000, 'ftol': 1e-14})
            if np.sum(res.x[2*N:]) > best_sum - 0.001:
                c_curr = res.x[:2*N].reshape(N, 2)
                r_curr = res.x[2*N:]
                s_curr = np.sum(r_curr)
                if s_curr > best_sum:
                    best_sum = s_curr
                    best_c = c_curr.copy()
                    best_r = r_curr.copy()
        except Exception:
            pass
            
        # Perturbation loop to escape local minima
        for _ in range(20):
            noise = 0.003 * (0.9 ** (0.1 * _))
            c_pert = best_c.copy() + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert = compute_radii_fast(c_pert) * 0.995
            v_pert = np.concatenate([c_pert.flatten(), r_pert])
            
            try:
                res = minimize(joint_obj, v_pert, method='SLSQP', bounds=bounds_joint,
                               constraints=cons_joint, options={'maxiter': 8000, 'ftol': 1e-14})
                if np.sum(res.x[2*N:]) > best_sum:
                    best_c = res.x[:2*N].reshape(N, 2)
                    best_r = res.x[2*N:]
                    best_sum = np.sum(best_r)
            except Exception:
                pass
                
    # Phase 3: Final LP refinement on fixed best centers for exact radii
    if best_c is not None:
        best_r_lp = get_max_radii_lp(best_c)
        if np.sum(best_r_lp) > best_sum:
            best_r = best_r_lp
            best_sum = np.sum(best_r)
            
    # Phase 4: Strict Numerical Repair
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(60):
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
