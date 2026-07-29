# sol_000203 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 263f0241) state=51b65a05 sum of radii=0.529465 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective_and_grad(x, r):
    """
    Computes the penalty objective and its gradient for a given configuration x and target radius r.
    x: flattened array of shape (52,) representing centers for 26 circles.
    r: target radius (float).
    Returns: (objective_value, gradient_array)
    """
    n = 26
    centers = x.reshape(n, 2)
    obj = 0.0
    grad = np.zeros_like(x)
    
    # Pairwise circle-circle overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.dot(diff, diff))
            
            if dist < 2.0 * r:
                slack = 2.0 * r - dist
                obj += slack**2
                
                if dist > 1e-8:
                    factor = -2.0 * slack / dist
                    # Gradient w.r.t circle i
                    grad[i*2] += factor * diff[0]
                    grad[i*2+1] += factor * diff[1]
                    # Gradient w.r.t circle j
                    grad[j*2] -= factor * diff[0]
                    grad[j*2+1] -= factor * diff[1]
        
        # Circle-boundary penetration penalty
        for k in range(2):
            val = centers[i, k]
            # Lower boundary: x < r
            if val < r:
                slack = r - val
                obj += slack**2
                grad[i*2+k] -= 2.0 * slack
            # Upper boundary: x > 1-r
            if val > 1.0 - r:
                slack = val - (1.0 - r)
                obj += slack**2
                grad[i*2+k] += 2.0 * slack
                
    return obj, grad

def run_packing():
    """
    Optimizes the positions of 26 circles in a unit square to maximize the sum of radii.
    Returns: (centers, radii, sum_radii)
    """
    np.random.seed(42)
    n = 26
    
    # Initial configuration: random points spread in the center to avoid boundary issues initially
    centers = np.random.rand(n, 2) * 0.4 + 0.3
    x0 = centers.flatten()
    
    # Optimization parameters
    r = 0.05
    step = 0.004
    max_iters = 400
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Iteratively increase radius and optimize positions
    for _ in range(max_iters):
        res = minimize(
            compute_objective_and_grad, 
            x0, 
            args=(r,),
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x0 = res.x
        
        # If packing is valid (low penalty), increase radius
        if res.fun < 1e-5:
            r += step
        else:
            # If stuck, slow down or stop
            if res.fun > 0.1:
                break
            r += step * 0.1
            
    centers_opt = x0.reshape(n, 2)
    
    # Compute the actual feasible radius from the optimized configuration
    min_sep = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers_opt[i] - centers_opt[j])
            if d < min_sep:
                min_sep = d
        for k in range(2):
            d_left = centers_opt[i, k]
            d_right = 1.0 - centers_opt[i, k]
            if d_left < min_sep:
                min_sep = d_left
            if d_right < min_sep:
                min_sep = d_right
                
    # Apply safety margin to guarantee validity against 1e-12 tolerance
    r_final = min_sep / 2.0 * 0.99999
    radii = np.full(n, r_final)
    sum_radii = np.sum(radii)
    
    return centers_opt, radii, sum_radii
