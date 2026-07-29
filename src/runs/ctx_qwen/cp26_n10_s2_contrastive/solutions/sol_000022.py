# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state c2ddf6ac) state=1df0edcb sum of radii=2.626337 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
PAIR_INDICES = np.triu_indices(N_CIRCLES, k=1)

def objective(params, n):
    """Negative sum of radii (to be minimized)."""
    return -np.sum(params[2*n:3*n])

def constraints(params, n):
    """
    Returns array of constraint values (must be >= 0).
    Includes boundary and pairwise non-overlap constraints.
    """
    cx = params[:n]
    cy = params[n:2*n]
    r = params[2*n:3*n]
    
    c = []
    # Boundary constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    dx = cx[PAIR_INDICES[0]] - cx[PAIR_INDICES[1]]
    dy = cy[PAIR_INDICES[0]] - cy[PAIR_INDICES[1]]
    dist = np.hypot(dx, dy)
    r_sum = r[PAIR_INDICES[0]] + r[PAIR_INDICES[1]]
    c.append(dist - r_sum)
    
    return np.concatenate(c)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_sum = -1.0
    best_params = None
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Topological configurations for row-based hexagonal packing
    configs = [
        [(6, 0.0), (5, 0.5), (6, 0.0), (5, 0.5), (4, 0.0)],
        [(5, 0.0), (6, 0.5), (5, 0.0), (6, 0.5), (4, 0.0)],
        [(7, 0.0), (6, 0.5), (6, 0.0), (7, 0.5)],
        [(5, 0.0), (5, 0.5), (6, 0.0), (5, 0.5), (5, 0.0)],
        [(4, 0.0), (6, 0.5), (6, 0.0), (6, 0.5), (4, 0.0)],
    ]
    
    np.random.seed(42)
    
    # Phase 1: Coarse search with multiple topologies and perturbations
    for trial in range(40):
        cfg = configs[trial % len(configs)]
        centers = np.zeros((n, 2))
        idx = 0
        num_rows = len(cfg)
        for r_idx, (count, shift) in enumerate(cfg):
            y = (r_idx + 1.0) / (num_rows + 1.0)
            x_spacing = 1.0 / (count + 1.0)
            for c_idx in range(count):
                x = (c_idx + 1.0) * x_spacing + shift * x_spacing
                centers[idx] = [x, y]
                idx += 1
                
        noise_scale = 0.015 + 0.005 * (trial % 6)
        centers += np.random.randn(n, 2) * noise_scale
        centers = np.clip(centers, 0.05, 0.95)
        
        p0 = np.concatenate([centers[:, 0], centers[:, 1], np.full(n, 0.035)])
        
        try:
            res = minimize(
                objective, p0, args=(n,), method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success and -res.fun > best_sum:
                best_sum = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Refinement around the best solution found
    if best_params is not None:
        for refine_trial in range(15):
            p0 = best_params.copy()
            p0[:2*n] += np.random.randn(2*n) * 0.004
            p0[2*n:] += np.random.randn(n) * 0.0015
            p0[:n] = np.clip(p0[:n], 0.0, 1.0)
            p0[n:2*n] = np.clip(p0[n:2*n], 0.0, 1.0)
            
            try:
                res = minimize(
                    objective, p0, args=(n,), method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                    options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
                )
                
                if res.success and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Fallback if optimization unexpectedly fails
    if best_params is None:
        best_params = np.zeros(3*n)
        for i in range(n):
            best_params[i] = 0.1 + (i % 5) * 0.2
            best_params[n+i] = 0.1 + (i // 5) * 0.2
        best_params[2*n:] = 0.03

    cx = best_params[:n]
    cy = best_params[n:2*n]
    r = best_params[2*n:3*n]
    
    # Post-processing to guarantee strict validity within validation tolerance
    # 1. Fix boundary violations
    for i in range(n):
        max_r = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if r[i] > max_r:
            r[i] = max(0.0, max_r - 1e-10)
            
    # 2. Fix overlap violations iteratively
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j] - 1e-10:
                    exc = r[i] + r[j] - d
                    r[i] -= exc / 2.0
                    r[j] -= exc / 2.0
                    changed = True
        if not changed:
            break
            
    r = np.maximum(r, 0.0)
    centers = np.column_stack((cx, cy))
    
    return centers, r, np.sum(r)
