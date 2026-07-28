# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=b095e67a sum of radii=2.339977 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def penalty_objective(vars, n, mu):
    """
    Computes the objective: negative sum of radii + penalty for constraint violations.
    """
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary penalties: circles must stay within [0, 1]
    # Left: x - r >= 0  => r - x <= 0
    v = r - c[:, 0]
    pen += np.sum(np.maximum(0, v)**2)
    # Right: 1 - x - r >= 0 => x + r - 1 <= 0
    v = c[:, 0] + r - 1.0
    pen += np.sum(np.maximum(0, v)**2)
    # Bottom: y - r >= 0 => r - y <= 0
    v = r - c[:, 1]
    pen += np.sum(np.maximum(0, v)**2)
    # Top: 1 - y - r >= 0 => y + r - 1 <= 0
    v = c[:, 1] + r - 1.0
    pen += np.sum(np.maximum(0, v)**2)
    
    # Overlap penalties: dist(i,j) >= r_i + r_j  => r_i + r_j - dist <= 0
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    viol = r_sum - dists
    i_idx, j_idx = np.triu_indices(n, k=1)
    pen += np.sum(np.maximum(0, viol[i_idx, j_idx])**2)
    
    return obj + mu * penalty

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-5, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    starts = []
    
    # 1. Hexagonal patterns with various row distributions
    row_counts_list = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4],
        [6, 5, 5, 6, 4], [5, 5, 5, 6, 5]
    ]
    for rc in row_counts_list:
        for r0 in [0.09, 0.10, 0.11]:
            pts = []
            y = r0
            row_idx = 0
            for cnt in rc:
                shift = r0 if row_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(pts) < n:
                        pts.append([x, y])
                    x += 2 * r0
                y += np.sqrt(3) * r0
                row_idx += 1
            pts = np.array(pts[:n])
            # Normalize to fit comfortably inside [0.05, 0.95]
            pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0)) * 0.7 + 0.15
            
            for _ in range(4):
                noise = np.random.uniform(-0.025, 0.025, pts.shape)
                p = np.clip(pts + noise, 0.05, 0.95)
                v = np.zeros(3*n)
                v[:2*n] = p.flatten()
                v[2*n:] = 0.095 + np.random.uniform(-0.01, 0.01)
                starts.append(v)
                
    # 2. Grid + center perturbations
    for _ in range(6):
        p = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        p += np.random.uniform(-0.03, 0.03, p.shape)
        p = np.clip(p, 0.05, 0.95)
        v = np.zeros(3*n)
        v[:2*n] = p.flatten()
        v[2*n:] = 0.09
        starts.append(v)
        
    # 3. Random starts
    for _ in range(6):
        v = np.zeros(3*n)
        p = np.random.uniform(0.15, 0.85, (n, 2))
        v[:2*n] = p.flatten()
        v[2*n:] = 0.08
        starts.append(v)
        
    # Continuation method for penalty parameter
    mus = [200, 2000, 20000]
    
    for start in starts:
        x = start.copy()
        try:
            for mu in mus:
                res = minimize(penalty_objective, x, args=(n, mu), method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 10000, 'ftol': 1e-14})
                x = res.x
                
            c_opt = x[:2*n].reshape(n, 2)
            r_opt = x[2*n:]
            
            # Strict validity check
            valid = True
            if np.any(np.isnan(c_opt)) or np.any(np.isnan(r_opt)): valid = False
            if np.any(r_opt < 1e-7): valid = False
            
            if valid:
                if np.any(c_opt[:,0] < r_opt - 1e-9) or np.any(c_opt[:,0] > 1 - r_opt + 1e-9): valid = False
                if np.any(c_opt[:,1] < r_opt - 1e-9) or np.any(c_opt[:,1] > 1 - r_opt + 1e-9): valid = False
                
                dists = np.linalg.norm(c_opt[:, None] - c_opt[None, :], axis=2)
                r_sums = r_opt[:, None] + r_opt[None, :]
                mask = np.triu(np.ones((n,n), dtype=bool), k=1)
                if np.any(dists[mask] < r_sums[mask] - 1e-9): valid = False
                
            if valid:
                s = np.sum(r_opt)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Fallback configuration if optimization yields nothing valid
    if best_centers is None:
        r_f = 0.09
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, r_f)
        k = 0
        y = r_f
        row = 0
        while k < n:
            shift = r_f if row % 2 == 1 else 0.0
            x = r_f + shift
            while x + r_f <= 1.0 and k < n:
                best_centers[k] = [x, y]
                k += 1
                x += 2 * r_f
            y += np.sqrt(3) * r_f
            row += 1
        best_sum = np.sum(best_radii)
        
    # Safety scaling to guarantee strict validity against 1e-12 tolerance
    best_radii *= 0.99999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
