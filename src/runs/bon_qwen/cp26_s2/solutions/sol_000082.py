# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3e058973) state=d5433efd sum of radii=0.189534 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss_and_grad(vars, N, lam):
    """Compute objective and gradient for penalty-based circle packing."""
    centers = vars.reshape(N, 3)[:, :2]
    radii = vars.reshape(N, 3)[:, 2]
    
    loss = -np.sum(radii)
    grad = np.zeros_like(vars)
    grad_r = -np.ones(N)
    
    # Pairwise overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            gap = radii[i] + radii[j] - dist
            
            if gap > 1e-9:
                factor = 2.0 * lam * gap
                inv_dist = 1.0 / (dist + 1e-12)
                
                # Gradient w.r.t centers
                grad[3*i]     += factor * dx * inv_dist
                grad[3*i + 1] += factor * dy * inv_dist
                grad[3*j]     -= factor * dx * inv_dist
                grad[3*j + 1] -= factor * dy * inv_dist
                
                # Gradient w.r.t radii
                grad_r[i] += factor
                grad_r[j] += factor
                
                loss += lam * gap * gap
                
        # Boundary constraints
        r = radii[i]
        x, y = centers[i, 0], centers[i, 1]
        
        # x < r  => r - x > 0
        gap = r - x
        if gap > 1e-9:
            factor = 2.0 * lam * gap
            grad_r[i] += factor
            grad[3*i] -= factor
            loss += lam * gap * gap
            
        # x > 1-r => x + r - 1 > 0
        gap = x + r - 1.0
        if gap > 1e-9:
            factor = 2.0 * lam * gap
            grad_r[i] += factor
            grad[3*i] += factor
            loss += lam * gap * gap
            
        # y < r
        gap = r - y
        if gap > 1e-9:
            factor = 2.0 * lam * gap
            grad_r[i] += factor
            grad[3*i + 1] -= factor
            loss += lam * gap * gap
            
        # y > 1-r
        gap = y + r - 1.0
        if gap > 1e-9:
            factor = 2.0 * lam * gap
            grad_r[i] += factor
            grad[3*i + 1] += factor
            loss += lam * gap * gap

    # Map radii gradients back to flat array
    grad[2::3] += grad_r
    return loss, grad

def shrink_to_feasible(centers, radii):
    """Scale radii down uniformly until all constraints are satisfied."""
    N = len(radii)
    scale = 1.0
    
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if sum_r > 1e-12:
                scale = min(scale, dist / sum_r)
                
    return radii * scale

def run_packing():
    N = 26
    lam = 8000.0
    bounds = [(1e-5, 1.0 - 1e-5), (1e-5, 1.0 - 1e-5), (1e-5, 0.5)] * N
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    num_trials = 40
    
    for trial in range(num_trials):
        # Mix of random and hex-grid initializations
        if trial < 10:
            # Hexagonal-ish grid
            c_init = np.zeros((N, 2))
            idx = 0
            y = 0.18
            row = 0
            while idx < N:
                x = 0.18 + (0.5 * (row % 2)) * 0.18
                while x < 0.82 and idx < N:
                    c_init[idx] = [x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)]
                    idx += 1
                    x += 0.18
                y += 0.15
                row += 1
        else:
            c_init = rng.uniform(0.2, 0.8, (N, 2))
            
        r_init = 0.08 + rng.uniform(-0.005, 0.005, N)
        
        x0 = np.empty(N * 3)
        for i in range(N):
            x0[3*i] = c_init[i, 0]
            x0[3*i + 1] = c_init[i, 1]
            x0[3*i + 2] = r_init[i]
            
        res = minimize(compute_loss_and_grad, x0, args=(N, lam), method='L-BFGS-B', 
                       jac=True, bounds=bounds, options={'maxiter': 4000, 'ftol': 1e-15})
        
        c_opt = res.x.reshape(N, 3)[:, :2]
        r_opt = res.x.reshape(N, 3)[:, 2]
        
        # Project to feasible region
        r_feas = shrink_to_feasible(c_opt, r_opt)
        
        # Strict validation check
        valid = True
        for i in range(N):
            if c_opt[i,0] < 1e-9 or c_opt[i,0] > 1-1e-9 or c_opt[i,1] < 1e-9 or c_opt[i,1] > 1-1e-9:
                valid = False; break
            if r_feas[i] < 1e-9:
                valid = False; break
                
        if valid:
            s = np.sum(r_feas)
            if s > best_sum:
                best_sum = s
                best_centers = c_opt.copy()
                best_radii = r_feas.copy()
                
    # Final uniform expansion search to push constraints tight
    if best_radii is not None:
        lo, hi = 1.0, 1.15
        for _ in range(25):
            mid = (lo + hi) / 2.0
            test_r = best_radii * mid
            is_valid = True
            
            # Boundary check
            for i in range(N):
                if best_centers[i,0] - test_r[i] < -1e-12 or best_centers[i,0] + test_r[i] > 1+1e-12 or \
                   best_centers[i,1] - test_r[i] < -1e-12 or best_centers[i,1] + test_r[i] > 1+1e-12:
                    is_valid = False; break
            if not is_valid:
                hi = mid
                continue
                
            # Overlap check
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                    if dist < test_r[i] + test_r[j] - 1e-12:
                        is_valid = False; break
                if not is_valid:
                    break
                    
            if is_valid:
                lo = mid
                best_radii = test_r
                best_sum = np.sum(test_r)
            else:
                hi = mid
                
    return best_centers, best_radii, best_sum
