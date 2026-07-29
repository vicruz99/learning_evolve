# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 556f0961) state=58c2cfd7 sum of radii=2.602618 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def penalty_grad(params, n=26):
    """Compute penalty value and its gradient for constraint violations."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    grad_pen = np.zeros_like(params)
    
    # Boundary constraints: circles must stay inside [0,1]^2
    for i in range(n):
        # x >= r
        v = r[i] - x[i]
        if v > 0:
            penalty += v**2
            grad_pen[i*3] -= 2.0 * v
            grad_pen[i*3+2] += 2.0 * v
        # x <= 1-r  =>  r + x - 1 <= 0
        v = r[i] + x[i] - 1.0
        if v > 0:
            penalty += v**2
            grad_pen[i*3] += 2.0 * v
            grad_pen[i*3+2] += 2.0 * v
        # y >= r
        v = r[i] - y[i]
        if v > 0:
            penalty += v**2
            grad_pen[i*3+1] -= 2.0 * v
            grad_pen[i*3+2] += 2.0 * v
        # y <= 1-r  =>  r + y - 1 <= 0
        v = r[i] + y[i] - 1.0
        if v > 0:
            penalty += v**2
            grad_pen[i*3+1] += 2.0 * v
            grad_pen[i*3+2] += 2.0 * v

    # Overlap constraints: distance >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                continue
            v = r[i] + r[j] - dist
            if v > 0:
                penalty += v**2
                term = 2.0 * v / dist
                grad_pen[i*3] -= term * dx
                grad_pen[i*3+1] -= term * dy
                grad_pen[j*3] += term * dx
                grad_pen[j*3+1] += term * dy
                grad_pen[i*3+2] += 2.0 * v
                grad_pen[j*3+2] += 2.0 * v
                
    return penalty, grad_pen

def obj(params, lam):
    """Objective function: negative sum of radii + lambda * penalty."""
    r = params[2::3]
    p, _ = penalty_grad(params)
    return -np.sum(r) + lam * p

def jac(params, lam):
    """Gradient of the objective function."""
    _, gp = penalty_grad(params)
    g = np.zeros_like(params)
    g[2::3] = -1.0  # gradient of -sum(r)
    return g + lam * gp

def init_params(n, seed=0):
    """Initialize circle positions in a grid pattern with small noise."""
    np.random.seed(seed)
    params = np.zeros(3 * n)
    cols = 6
    rows = 5
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= n: break
            params[idx*3] = (c_idx + 0.5) / cols + np.random.normal(0, 0.02)
            params[idx*3+1] = (r_idx + 0.5) / rows + np.random.normal(0, 0.02)
            params[idx*3+2] = 0.03
            idx += 1
    return params

def run_packing():
    n = 26
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    best_params = None
    best_sum = -1.0
    
    # Run multiple restarts to avoid local minima
    for seed in range(5):
        params = init_params(n, seed)
        
        # Sequentially increase penalty weight to enforce constraints tightly
        lams = [10, 100, 1000, 5000, 20000]
        for lam in lams:
            res = minimize(obj, params, jac=jac, args=(lam,), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-10})
            params = res.x
            
        # Evaluate current configuration
        r_final = params[2::3]
        current_sum = np.sum(r_final)
        if current_sum > best_sum:
            best_sum = current_sum
            best_params = params.copy()
            
    # Extract results
    centers = np.column_stack((best_params[0::3], best_params[1::3]))
    radii = best_params[2::3].copy()
    
    # Post-processing: ensure strict validity within tolerance
    # Uniformly shrink radii if any constraint is violated
    scale = 1.0
    for _ in range(20):
        valid = True
        # Check boundaries
        for i in range(n):
            cx, cy, cr = centers[i,0], centers[i,1], radii[i]
            if cx - cr < -1e-12 or cx + cr > 1 + 1e-12 or cy - cr < -1e-12 or cy + cr > 1 + 1e-12:
                valid = False
                break
        # Check overlaps
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if dist < radii[i] + radii[j] - 1e-12:
                        valid = False
                        break
                if not valid: break
                
        if valid:
            break
        scale *= 0.995
        radii *= scale
        
    return centers, radii, np.sum(radii)
