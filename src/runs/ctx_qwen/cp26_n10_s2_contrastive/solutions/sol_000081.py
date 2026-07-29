# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000051 (state 921aef56) state=98a7c0f5 sum of radii=2.594643 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints g(x) >= 0:
    - Squared pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    - Boundary clearance: x >= r, 1-x >= r, y >= r, 1-y >= r
    Squared distances provide smooth gradients for SLSQP.
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    c_overlap = dx**2 + dy**2 - (r[I] + r[J])**2
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_overlap, c_bound])

def solve_radii_lp(centers):
    """Given fixed centers, find radii that maximize sum via LP."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((n*(n-1)//2, n))
    b_ub = np.zeros(n*(n-1)//2)
    
    # Precompute all pairwise distances
    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :])**2).sum(axis=2))
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    bounds_r = []
    for i in range(n):
        ub = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds_r.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
        
    return np.zeros(n), 0.0

def init_hex():
    """Generate a structured hexagonal lattice initialization."""
    rng = np.random.RandomState(0)
    counts = [6, 5, 6, 5, 4]  # Sums to 26
    s = 0.16
    y_step = s * np.sqrt(3) / 2
    centers = []
    y = s/2
    for r_idx, c in enumerate(counts):
        x_start = s/2 + (r_idx % 2) * s/2
        for _ in range(c):
            centers.append([x_start + _*s, y])
        y += y_step
        
    centers = np.array(centers[:26])
    cx, cy = centers[:,0], centers[:,1]
    
    # Normalize to fit comfortably inside [0.05, 0.95]
    centers[:,0] = 0.05 + (cx - cx.min()) / (cx.max() - cx.min() + 1e-9) * 0.9
    centers[:,1] = 0.05 + (cy - cy.min()) / (cy.max() - cy.min() + 1e-9) * 0.9
    
    # Break symmetry with tiny noise
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c, best_r = None, None
    
    # --- Phase 1: Broad Multi-Start Search ---
    inits = [init_hex()]
    rng = np.random.RandomState(0)
    for s in range(25):
        c = rng.uniform(0.1, 0.9, (N, 2))
        inits.append(c)
        
    for c0 in inits:
        r0, _ = solve_radii_lp(c0)
        r0 = np.maximum(r0, 1e-5)
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:,0]
        x0[1::3] = c0[:,1]
        x0[2::3] = r0 * 0.95  # Slight shrink to guarantee initial feasibility
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_lp, s_lp = solve_radii_lp(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c, best_r = c_opt.copy(), r_lp.copy()
        except Exception:
            pass

    # --- Phase 2: Basin Hopping / Local Perturbation ---
    if best_c is not None:
        rng = np.random.RandomState(42)
        for trial in range(60):
            # Decaying noise schedule
            scale = 0.005 * np.exp(-trial/30.0) + 0.0005
            c_p = best_c + rng.normal(0, scale, best_c.shape)
            c_p = np.clip(c_p, 0.01, 0.99)
            
            r_p, _ = solve_radii_lp(c_p)
            r_p = np.maximum(r_p, 1e-5)
            x_p = np.zeros(3*N)
            x_p[0::3] = c_p[:,0]
            x_p[1::3] = c_p[:,1]
            x_p[2::3] = r_p * 0.98
            
            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_lp, s_lp = solve_radii_lp(c_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_c, best_r = c_opt.copy(), r_lp.copy()
            except Exception:
                pass
                
    # --- Phase 3: Radius Relaxation & Re-expansion ---
    # Helps escape configurations where circles are tightly locked against boundaries
    if best_c is not None:
        rng = np.random.RandomState(99)
        for _ in range(20):
            c_rel = best_c + rng.normal(0, 0.008, best_c.shape)
            c_rel = np.clip(c_rel, 0.02, 0.98)
            r_rel, _ = solve_radii_lp(c_rel)
            r_rel = np.maximum(r_rel * 0.85, 1e-5)  # Shrink to create slack
            
            x_rel = np.zeros(3*N)
            x_rel[0::3] = c_rel[:,0]
            x_rel[1::3] = c_rel[:,1]
            x_rel[2::3] = r_rel
            
            try:
                res = minimize(objective, x_rel, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_lp, s_lp = solve_radii_lp(c_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_c, best_r = c_opt.copy(), r_lp.copy()
            except Exception:
                pass

    # Fallback safety net
    if best_c is None:
        best_c = init_hex()
        best_r, best_sum = solve_radii_lp(best_c)
        
    # --- Phase 4: Strict Validation & Numerical Fixing ---
    r = best_r.copy()
    
    # Enforce hard boundary limits
    for i in range(N):
        mx = min(best_c[i,0], 1.0-best_c[i,0], best_c[i,1], 1.0-best_c[i,1])
        r[i] = min(r[i], max(0.0, mx - 1e-9))
        
    # Iteratively resolve any residual overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                if d < r[i] + r[j] - 1e-9:
                    exc = r[i] + r[j] - d
                    r[i] -= exc/2.0
                    r[j] -= exc/2.0
                    changed = True
        if not changed:
            break
            
    r = np.maximum(r, 0.0)
    return best_c, r, float(np.sum(r))
