# sol_000314 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000255 (state 9c9fca7e) state=faeb0d1e sum of radii=2.419880 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def build_A(n):
    """Builds the constant inequality matrix A_ub for the LP radius solver."""
    m = 4*n + n*(n-1)//2
    A = np.zeros((m, n))
    k = 0
    for i in range(n):
        A[k,i] = 1.0; k+=1
        A[k,i] = 1.0; k+=1
        A[k,i] = 1.0; k+=1
        A[k,i] = 1.0; k+=1
    for i in range(n):
        for j in range(i+1, n):
            A[k,i] = 1.0; A[k,j] = 1.0; k+=1
    return A

def solve_lp_and_grad(centers, A_ub, n):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, sum_radii, and gradient of sum_radii w.r.t centers using LP duals.
    """
    m = A_ub.shape[0]
    b = np.zeros(m)
    k = 0
    for i in range(n):
        x, y = centers[i]
        b[k] = x; k += 1
        b[k] = 1.0 - x; k += 1
        b[k] = y; k += 1
        b[k] = 1.0 - y; k += 1
        
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            b[k] = np.hypot(dx, dy)
            k += 1
            
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b, bounds=(0, None), method='highs')
    
    if not res.success:
        # Fallback to safe geometric radii if LP fails
        lims = np.minimum(np.minimum(centers[:,0], 1-centers[:,0]), 
                          np.minimum(centers[:,1], 1-centers[:,1]))
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists, axis=1) / 2.0
        r_fb = np.minimum(lims, min_d) * 0.9
        return r_fb, np.sum(r_fb), np.zeros((n, 2))
        
    radii = res.x
    obj = -res.fun
    
    # Extract dual variables (marginals) for gradient computation
    try:
        marg = np.asarray(res.marginals.ineqlin)
    except (AttributeError, TypeError):
        marg = np.zeros(m)
        
    grad = np.zeros((n, 2))
    k = 0
    # Boundary constraints gradients
    for i in range(n):
        grad[i, 0] -= marg[k]; k+=1  # x constraint
        grad[i, 0] += marg[k]; k+=1  # 1-x constraint
        grad[i, 1] -= marg[k]; k+=1  # y constraint
        grad[i, 1] += marg[k]; k+=1  # 1-y constraint
        
    # Pairwise distance constraints gradients
    for i in range(n):
        for j in range(i+1, n):
            mu = marg[k]
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            d = b[k]
            if d > 1e-12:
                fx = -mu * dx / d
                fy = -mu * dy / d
                grad[i, 0] += fx
                grad[i, 1] += fy
                grad[j, 0] -= fx
                grad[j, 1] -= fy
            k += 1
    return radii, obj, grad

def obj_jac(c_flat, n, A_ub):
    """Objective and Jacobian for L-BFGS-B: minimize negative sum of radii."""
    centers = c_flat.reshape(n, 2)
    radii, obj, grad = solve_lp_and_grad(centers, A_ub, n)
    return -obj, -grad.flatten()

def run_packing():
    n = 26
    A_ub = build_A(n)
    bounds_c = [(0.005, 0.995)] * (2*n)
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    configs = []
    # 1. Hexagonal lattice starts with varying densities
    for r0 in [0.07, 0.08, 0.09, 0.10, 0.11]:
        pts = []
        y = r0
        row = 0
        while len(pts) < n:
            shift = r0 if row % 2 == 1 else 0.0
            x = r0 + shift
            while x <= 1.0 - r0 and len(pts) < n:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
        configs.append(np.array(pts[:n]))
        
    # 2. Rotated hexagonal starts to break symmetries
    for cfg in configs[:3]:
        for angle in [0.1, -0.1, 0.2, -0.2]:
            c = cfg - 0.5
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            r_cfg = c @ rot.T + 0.5
            # Normalize to fit comfortably in [0.05, 0.95]
            r_cfg = (r_cfg - r_cfg.min(axis=0)) / (r_cfg.max(axis=0) - r_cfg.min(axis=0)) * 0.9 + 0.05
            configs.append(r_cfg)
            
    # 3. Random dense starts
    for _ in range(12):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        cfg = np.clip(cfg, 0.01, 0.99)
        try:
            res = minimize(obj_jac, cfg.flatten(), args=(n, A_ub), method='L-BFGS-B', 
                           jac=True, bounds=bounds_c, options={'maxiter': 4000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(n, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt, A_ub, n)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation to escape local minima
    if best_centers is not None:
        for _ in range(150):
            idx = rng.integers(n)
            old = best_centers[idx].copy()
            best_centers[idx] = np.clip(old + rng.uniform(-0.006, 0.006, 2), 0.01, 0.99)
            r_p, s_p, _ = solve_lp_and_grad(best_centers, A_ub, n)
            if s_p > best_sum + 1e-8:
                best_sum = s_p
                best_radii = r_p.copy()
            else:
                best_centers[idx] = old
                
        # Phase 3: Jitter & Re-optimize from perturbed states
        for _ in range(20):
            c_jit = best_centers + rng.uniform(-0.003, 0.003, best_centers.shape)
            c_jit = np.clip(c_jit, 0.01, 0.99)
            try:
                res_j = minimize(obj_jac, c_jit.flatten(), args=(n, A_ub), method='L-BFGS-B', 
                                 jac=True, bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-13})
                if np.isfinite(res_j.fun):
                    c_opt = res_j.x.reshape(n, 2)
                    r_opt, s_opt, _ = solve_lp_and_grad(c_opt, A_ub, n)
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
            except Exception:
                continue

    # Fallback configuration
    if best_centers is None:
        best_centers = configs[0]
        best_radii, best_sum, _ = solve_lp_and_grad(best_centers, A_ub, n)
        
    # Final strict numerical scaling to guarantee validator tolerance (1e-12)
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i,0], best_centers[i,1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
