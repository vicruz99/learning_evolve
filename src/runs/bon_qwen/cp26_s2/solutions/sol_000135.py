# sol_000135 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc363b95) state=c51846d7 sum of radii=2.619980 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss_and_grad(vars, n, alpha):
    """
    Computes the loss and its gradient for the circle packing optimization.
    Loss = -sum(radii) + penalty for overlaps and boundary violations.
    """
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    loss = -np.sum(r)
    grad = np.zeros_like(vars)
    grad[2*n:] = -1.0  # Gradient of -sum(r) w.r.t radii
    
    # Pairwise overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            diff = c[i] - c[j]
            dist = np.sqrt(np.sum(diff**2)) + 1e-12
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                loss += alpha * overlap**2
                dir_ = diff / dist
                # Gradient w.r.t centers
                grad[2*i:2*i+2] -= 2 * alpha * overlap * dir_
                grad[2*j:2*j+2] += 2 * alpha * overlap * dir_
                # Gradient w.r.t radii
                grad[2*n + i] += 2 * alpha * overlap
                grad[2*n + j] += 2 * alpha * overlap
                
    # Boundary containment penalties
    for i in range(n):
        # Left boundary: x >= r
        gap = r[i] - c[i, 0]
        if gap > 0:
            loss += alpha * gap**2
            grad[2*i] -= 2 * alpha * gap
            grad[2*n + i] += 2 * alpha * gap
            
        # Right boundary: 1-x >= r
        gap = r[i] + c[i, 0] - 1.0
        if gap > 0:
            loss += alpha * gap**2
            grad[2*i] += 2 * alpha * gap
            grad[2*n + i] += 2 * alpha * gap
            
        # Bottom boundary: y >= r
        gap = r[i] - c[i, 1]
        if gap > 0:
            loss += alpha * gap**2
            grad[2*i+1] -= 2 * alpha * gap
            grad[2*n + i] += 2 * alpha * gap
            
        # Top boundary: 1-y >= r
        gap = r[i] + c[i, 1] - 1.0
        if gap > 0:
            loss += alpha * gap**2
            grad[2*i+1] += 2 * alpha * gap
            grad[2*n + i] += 2 * alpha * gap
            
    return loss, grad

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Initialize with a 5x5 grid + 1 central circle
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i*0.2, 0.1 + j*0.2])
    centers.append([0.5, 0.5])
    centers = np.array(centers)
    radii = np.ones(n) * 0.1
    radii[-1] = 0.01  # Smaller initial radius for the 26th circle
    
    # Concatenate centers and radii into a single optimization variable vector
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Add small perturbation to break symmetry and aid optimization
    x0 += np.random.randn(len(x0)) * 0.002
    x0[:2*n] = np.clip(x0[:2*n], 0.0, 1.0)
    x0[2*n:] = np.clip(x0[2*n:], 0.0, 0.5)
    
    # Define bounds for centers [0,1] and radii [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Run numerical optimization
    res = minimize(compute_loss_and_grad, x0, jac=True, method='L-BFGS-B', 
                   bounds=bounds, args=(n, 2000.0),
                   options={'maxiter': 8000, 'ftol': 1e-14, 'gtol': 1e-12})
                   
    centers_opt = res.x[:2*n].reshape(n, 2)
    radii_opt = res.x[2*n:]
    
    # Strict boundary enforcement
    for i in range(n):
        r_val = radii_opt[i]
        x, y = centers_opt[i]
        r_val = min(r_val, x, 1.0-x, y, 1.0-y)
        radii_opt[i] = max(r_val, 0.0)
        
    # Strict pairwise overlap enforcement
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
            r_sum = radii_opt[i] + radii_opt[j]
            if r_sum > dist - 1e-12:
                shrink = (r_sum - dist + 1e-12) / 2.0 + 1e-12
                radii_opt[i] = max(0.0, radii_opt[i] - shrink)
                radii_opt[j] = max(0.0, radii_opt[j] - shrink)
                
    return centers_opt, radii_opt, float(np.sum(radii_opt))
