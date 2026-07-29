# sol_000218 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 76d635d8) state=b6463518 sum of radii=2.596082 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        return False

    if np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True

def objective_and_grad(params, n):
    """
    Computes the negative sum of radii (to minimize) and gradient.
    Includes soft constraints for boundaries and overlaps.
    
    params: array of size 3*n. 
    Structure: [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i + 1]
        radii[i] = params[3*i + 2]
    
    # We want to maximize sum(radii), so minimize -sum(radii)
    score = -np.sum(radii)
    grad = np.zeros_like(params)
    
    # Gradient for -sum(radii) is -1 for each r component
    for i in range(n):
        grad[3*i + 2] = -1.0
        
    penalty = 0.0
    grad_penalty = np.zeros_like(params)
    
    # Large penalty coefficients
    K_bound = 100.0
    K_overlap = 100.0
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # Equivalent to: x - r >= 0, r - x + 1 >= 0 => 1 - x - r >= 0
    # y - r >= 0, 1 - y - r >= 0
    
    for i in range(n):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        
        # x - r >= 0
        viol = max(0.0, r - x)
        penalty += K_bound * viol**2
        if viol > 0:
            grad_penalty[3*i] += 2 * K_bound * viol * (-1) # d/dx (r-x)^2 = -2(r-x)
            grad_penalty[3*i+2] += 2 * K_bound * viol * (1) # d/dr (r-x)^2 = 2(r-x)
            
        # 1 - x - r >= 0
        viol = max(0.0, x + r - 1)
        penalty += K_bound * viol**2
        if viol > 0:
            grad_penalty[3*i] += 2 * K_bound * viol * (1)
            grad_penalty[3*i+2] += 2 * K_bound * viol * (1)
            
        # y - r >= 0
        viol = max(0.0, r - y)
        penalty += K_bound * viol**2
        if viol > 0:
            grad_penalty[3*i+1] += 2 * K_bound * viol * (-1)
            grad_penalty[3*i+2] += 2 * K_bound * viol * (1)
            
        # 1 - y - r >= 0
        viol = max(0.0, y + r - 1)
        penalty += K_bound * viol**2
        if viol > 0:
            grad_penalty[3*i+1] += 2 * K_bound * viol * (1)
            grad_penalty[3*i+2] += 2 * K_bound * viol * (1)
            
        # r >= 0
        if r < 0:
            penalty += K_bound * r**2
            grad_penalty[3*i+2] += 2 * K_bound * r

    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = params[3*i] - params[3*j]
            dy = params[3*i+1] - params[3*j+1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            r_sum = params[3*i+2] + params[3*j+2]
            
            if dist < r_sum:
                viol = r_sum - dist
                penalty += K_overlap * viol**2
                
                # Gradient w.r.t positions and radii
                # d(viol)/dx_i = - (dx/dist) * (-1)? 
                # viol = r_i + r_j - sqrt(dx^2 + dy^2)
                # d(viol)/dx_i = - (1/(2*dist)) * 2*dx * (-1) ? 
                # Wait, dx = x_i - x_j. d(dist)/dx_i = dx/dist.
                # d(viol)/dx_i = - dx/dist.
                # d(viol)/dx_j = dx/dist.
                
                if dist > 1e-9:
                    deriv_dist_x = dx / dist
                    deriv_dist_y = dy / dist
                else:
                    deriv_dist_x = 0.0
                    deriv_dist_y = 0.0
                
                # Gradient of penalty term 2*K*viol * d(viol)/dp
                factor = 2 * K_overlap * viol
                
                # w.r.t x_i
                grad_penalty[3*i] += factor * (-deriv_dist_x)
                # w.r.t y_i
                grad_penalty[3*i+1] += factor * (-deriv_dist_y)
                # w.r.t r_i
                grad_penalty[3*i+2] += factor * (1.0)
                
                # w.r.t x_j
                grad_penalty[3*j] += factor * (deriv_dist_x)
                # w.r.t y_j
                grad_penalty[3*j+1] += factor * (deriv_dist_y)
                # w.r.t r_j
                grad_penalty[3*j+2] += factor * (1.0)

    total_grad = grad + grad_penalty
    return score + penalty, total_grad

def run_packing():
    n = 26
    best_score = -np.inf
    best_params = None
    
    # Try multiple initializations
    # 1. Hexagonal packing initialization
    # 2. Random initialization
    
    for attempt in range(10):
        params = np.zeros(3 * n)
        
        if attempt == 0:
            # Hexagonal-ish grid
            # 5 rows
            # Row counts: 6, 5, 6, 5, 4 (sum 26)
            # Or 5, 6, 5, 6, 4?
            # Let's try to fit them roughly
            
            # Approximate radius 0.1
            r_est = 0.101
            rows = [6, 5, 6, 5, 4]
            idx = 0
            y_pos = r_est
            
            # To fit in unit square, scale down
            # Width needed for 6 circles: 12*r
            # Height needed: 2*r + 4*sqrt(3)*r
            
            # Let's just place them in a grid and let optimizer fix it
            # Simple grid 6x5
            cx = 0.1
            cy = 0.1
            step_x = 0.19
            step_y = 0.19
            
            cnt = 0
            for r in range(5):
                for c in range(6):
                    if cnt < n:
                        params[3*cnt] = cx + c * step_x + (r % 2) * (step_x / 2)
                        params[3*cnt+1] = cy + r * step_y
                        params[3*cnt+2] = 0.10
                        cnt += 1
        else:
            # Random initialization with small radii
            for i in range(n):
                params[3*i] = np.random.uniform(0.2, 0.8)
                params[3*i+1] = np.random.uniform(0.2, 0.8)
                params[3*i+2] = np.random.uniform(0.05, 0.15)
        
        # Optimization using a simple gradient descent or scipy
        # Since we defined gradient, we can try scipy.optimize.minimize
        try:
            from scipy.optimize import minimize
            # Use L-BFGS-B or similar, but our function is custom
            # Actually, L-BFGS-B requires bounds. We can set bounds [0,1] for x,y and [0,1] for r.
            bounds = [(0.0, 1.0)] * (3 * n)
            
            # To handle the non-convexity, we might need a robust method
            # But minimize with custom grad might work if close
            # Let's use a simple custom descent first to avoid scipy dependency issues if any, 
            # but scipy is allowed.
            
            res = minimize(objective_and_grad, params, args=(n,), method='L-BFGS-B', 
                           jac=True, bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            # Fallback simple gradient ascent if scipy fails
            lr = 0.001
            for step in range(500):
                val, grad = objective_and_grad(params, n)
                # Gradient descent on negative sum (so ascent on sum)
                # Wait, objective returns -sum + penalty. Minimizing it maximizes sum.
                # grad is d/dp (-sum + penalty). We want to move against grad.
                params = params - lr * grad
                lr *= 0.999
                
                # Clip
                for i in range(n):
                    params[3*i] = np.clip(params[3*i], 0, 1)
                    params[3*i+1] = np.clip(params[3*i+1], 0, 1)
                    params[3*i+2] = np.clip(params[3*i+2], 0, 0.5) # r < 0.5

                val, _ = objective_and_grad(params, n)
                if -val > best_score:
                     # Check validity roughly
                     pass 
                     # We can't check validity easily inside loop without overhead, 
                     # but penalty handles it.
                     best_score = -val
                     best_params = params.copy()

    if best_params is None:
        # Return a safe fallback
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.01
        centers[:, 0] = 0.5
        centers[:, 1] = 0.5
        return centers, radii, sum(radii)

    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_params[3*i]
        centers[i, 1] = best_params[3*i+1]
        radii[i] = best_params[3*i+2]
    
    # Final cleanup: ensure non-negative radii and clip to box
    radii = np.maximum(radii, 1e-9)
    for i in range(n):
        r = radii[i]
        x = centers[i, 0]
        y = centers[i, 1]
        # Adjust center if outside valid range for radius r
        # x must be in [r, 1-r]
        centers[i, 0] = np.clip(x, r, 1 - r)
        centers[i, 1] = np.clip(y, r, 1 - r)
        
    # Resolve overlaps by shrinking radii slightly if needed
    # Iterative resolution
    for _ in range(10):
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                req_dist = radii[i] + radii[j]
                if dist < req_dist - 1e-9:
                    # Overlap detected, reduce radii
                    # Split difference
                    diff = req_dist - dist
                    # Reduce both radii proportionally? Or just scale down
                    # Simple heuristic: reduce sum of radii to match distance
                    scale = dist / req_dist
                    radii[i] *= scale
                    radii[j] *= scale
    
    # Re-check boundaries after shrinking
    for i in range(n):
        r = radii[i]
        x = centers[i, 0]
        y = centers[i, 1]
        if x < r: radii[i] = x; centers[i, 0] = x # Actually x=r if x<r
        if x > 1 - r: radii[i] = 1 - x
        if y < r: radii[i] = y
        if y > 1 - r: radii[i] = 1 - y
        
    # Re-clip centers
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)

    sum_radii = np.sum(radii)
    
    # Validation check
    if not validate_packing(centers, radii):
        # If invalid, return a known valid simple packing
        # 5x5 grid r=0.1
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.099 # Slightly less to be safe
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n:
                    centers[idx, 0] = 0.1 + c * 0.2
                    centers[idx, 1] = 0.1 + r * 0.2
                    radii[idx] = 0.1 # 0.1 is exact for grid, but let's be safe
                    idx += 1
        # Fill remaining
        while idx < n:
            centers[idx, 0] = 0.5
            centers[idx, 1] = 0.5
            radii[idx] = 0.001
            idx += 1
        sum_radii = np.sum(radii)
        
    return centers, radii, sum_radii
