# sol_000291 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 86cff419) state=1235f2da sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss_and_grad(params, n=26, mu=1e6):
    """
    Computes the objective function and its gradient for the circle packing problem.
    Objective: Maximize sum of radii => Minimize -sum(r)
    Constraints handled via quadratic penalty terms.
    """
    pts = params.reshape(n, 3)
    loss = 0.0
    grad = np.zeros_like(params)
    
    # Objective: minimize -sum(r)
    loss -= np.sum(pts[:, 2])
    grad[:, 2] -= 1.0
    
    for i in range(n):
        xi, yi, ri = pts[i]
        
        # Boundary constraints: x - r >= 0
        if xi - ri < 0:
            pen = xi - ri
            loss += mu * pen**2
            grad[i, 0] += 2 * mu * pen
            grad[i, 2] -= 2 * mu * pen
        # Boundary constraints: x + r <= 1
        if xi + ri - 1 > 0:
            pen = xi + ri - 1
            loss += mu * pen**2
            grad[i, 0] += 2 * mu * pen
            grad[i, 2] += 2 * mu * pen
        # Boundary constraints: y - r >= 0
        if yi - ri < 0:
            pen = yi - ri
            loss += mu * pen**2
            grad[i, 1] += 2 * mu * pen
            grad[i, 2] -= 2 * mu * pen
        # Boundary constraints: y + r <= 1
        if yi + ri - 1 > 0:
            pen = yi + ri - 1
            loss += mu * pen**2
            grad[i, 1] += 2 * mu * pen
            grad[i, 2] += 2 * mu * pen
            
        for j in range(i + 1, n):
            xj, yj, rj = pts[j]
            dx = xi - xj
            dy = yi - yj
            d = np.hypot(dx, dy)
            if d < 1e-8:
                d = 1e-8
            sum_r = ri + rj
            overlap = sum_r - d
            if overlap > 0:
                loss += mu * overlap**2
                
                # Gradient w.r.t radii
                grad[i, 2] += 2 * mu * overlap
                grad[j, 2] += 2 * mu * overlap
                
                # Gradient w.r.t centers
                fx = -2 * mu * overlap * (dx / d)
                fy = -2 * mu * overlap * (dy / d)
                grad[i, 0] += fx
                grad[j, 0] -= fx
                grad[i, 1] += fy
                grad[j, 1] -= fy
                
    return loss, grad

def run_packing():
    n = 26
    best_params = None
    best_loss = np.inf
    
    # Multiple restarts from perturbed grid layouts
    for seed in range(10):
        np.random.seed(seed)
        pts = np.zeros((n, 3))
        cols = 5
        rows = int(np.ceil(n / cols))
        for i in range(n):
            r_idx = i // cols
            c_idx = i % cols
            # Initialize near a grid, add jitter
            pts[i, 0] = (c_idx + 0.5) / cols + np.random.normal(0, 0.03)
            pts[i, 1] = (r_idx + 0.5) / rows + np.random.normal(0, 0.03)
            pts[i, 2] = 0.05  # Start with small radii to avoid initial overlaps
            
        # Ensure initial positions are strictly inside
        pts[:, 0] = np.clip(pts[:, 0], 0.1, 0.9)
        pts[:, 1] = np.clip(pts[:, 1], 0.1, 0.9)
        
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * n
        
        try:
            # Stage 1: Coarse optimization to resolve overlaps and find feasible region
            res1 = minimize(compute_loss_and_grad, pts.flatten(), 
                            args=(n, 1e5), method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 1500})
            
            # Stage 2: Fine optimization with high penalty to strictly enforce constraints
            # and maximize radii
            res2 = minimize(compute_loss_and_grad, res1.x, 
                            args=(n, 1e8), method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 1500})
            
            if res2.fun < best_loss:
                best_loss = res2.fun
                best_params = res2.x.copy()
        except Exception:
            continue
            
    # Fallback to a valid, albeit suboptimal, grid configuration if optimization fails
    if best_params is None:
        pts = np.zeros((n, 3))
        for i in range(n):
            pts[i, 0] = (i % 5) * 0.15 + 0.1
            pts[i, 1] = (i // 5) * 0.15 + 0.1
            pts[i, 2] = 0.05
        best_params = pts.flatten()
        
    pts = best_params.reshape(n, 3)
    centers = pts[:, :2]
    radii = pts[:, 2]
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
