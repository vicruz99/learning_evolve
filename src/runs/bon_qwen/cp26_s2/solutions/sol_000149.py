# sol_000149 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=6202d2a2 sum of radii=2.583154 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars, n):
    """Negative sum of radii to minimize."""
    return -np.sum(vars[2*n:])

def constraints(vars, n):
    """Vectorized inequality constraints: >= 0."""
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    c = []
    # Boundary constraints
    c.append(centers[:, 0] - radii)
    c.append(1.0 - centers[:, 0] - radii)
    c.append(centers[:, 1] - radii)
    c.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c_pairs = d2[mask] - r_sum[mask]**2
    c.append(c_pairs)
    
    return np.concatenate(c)

def run_packing():
    n = 26
    best_sum = 0.0
    best_vars = None
    
    np.random.seed(42)
    for restart in range(5):
        # Initialize with a hexagonal-like pattern
        rows = [6, 5, 6, 5, 4]
        centers_init = []
        y = 0.15
        dy = 0.175
        dx = 0.16
        for i, count in enumerate(rows):
            for k in range(count):
                x = 0.15 + k * dx + (i % 2) * dx / 2
                centers_init.append([x, y])
            y += dy
        centers_init = np.array(centers_init)
        
        # Add small perturbation to break symmetry
        centers_init += np.random.randn(n, 2) * 0.005
        centers_init = np.clip(centers_init, 0.05, 0.95)
        
        radii_init = np.full(n, 0.085)
        vars_init = np.concatenate([centers_init.flatten(), radii_init])
        
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.2)] * n
        cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
        
        res = minimize(
            objective, 
            vars_init, 
            args=(n,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 5000, 'ftol': 1e-10}
        )
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_vars = res.x
            
    centers_out = best_vars[:2*n].reshape(n, 2)
    radii_out = best_vars[2*n:]
    
    # Robustness check: slightly shrink radii if any constraint is violated near tolerance
    max_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers_out[i] - centers_out[j])**2))
            if dist < radii_out[i] + radii_out[j] - 1e-10:
                overlap = (radii_out[i] + radii_out[j] - dist)
                ratio = overlap / (radii_out[i] + radii_out[j])
                if ratio > max_violation:
                    max_violation = ratio
                    
    if max_violation > 0:
        radii_out *= (1.0 - max_violation - 1e-8)
        
    return centers_out, radii_out, float(np.sum(radii_out))
