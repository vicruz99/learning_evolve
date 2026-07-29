# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1907b6e7) state=1a0b54ae sum of radii=2.625967 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective_and_grad(v, n, mu):
    """Computes the penalty-based objective and its analytical gradient."""
    X = v[:2 * n].reshape(n, 2)
    R = v[2 * n:]
    
    loss = -np.sum(R)
    grad_X = np.zeros_like(X)
    grad_R = -np.ones(n)
    
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            d = np.sqrt(np.dot(diff, diff))
            if d < 1e-12:
                d = 1e-12
                diff = np.array([1.0, 0.0])
            
            g = R[i] + R[j] - d
            if g > 0:
                loss += mu * g ** 2
                grad_R[i] += 2 * mu * g
                grad_R[j] += 2 * mu * g
                factor = -2.0 * mu * g / d
                grad_X[i] += factor * diff
                grad_X[j] -= factor * diff
                
    # Boundary constraints
    for i in range(n):
        for dim in range(2):
            # Left / Bottom boundary
            g1 = R[i] - X[i, dim]
            if g1 > 0:
                loss += mu * g1 ** 2
                grad_R[i] += 2 * mu * g1
                grad_X[i, dim] -= 2 * mu * g1
                
            # Right / Top boundary
            g2 = R[i] - (1.0 - X[i, dim])
            if g2 > 0:
                loss += mu * g2 ** 2
                grad_R[i] += 2 * mu * g2
                grad_X[i, dim] += 2 * mu * g2
                
    return loss, np.concatenate([grad_X.flatten(), grad_R])

def obj_fun(v, n, mu):
    return compute_objective_and_grad(v, n, mu)[0]

def obj_jac(v, n, mu):
    return compute_objective_and_grad(v, n, mu)[1]

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Initialize with a dense 5x5 grid + center, perturbed to break symmetry
    base_centers = []
    for r in range(5):
        for c in range(5):
            base_centers.append([0.1 + c * 0.2, 0.1 + r * 0.2])
    base_centers.append([0.5, 0.5])
    centers = np.array(base_centers)
    centers += np.random.normal(0, 0.005, size=centers.shape)
    centers = np.clip(centers, 0.1, 0.9)
    
    radii = np.full(n, 0.05)
    vars = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    best_vars = vars.copy()
    mu = 200.0
    
    # Sequential penalty optimization
    for step in range(30):
        res = minimize(obj_fun, best_vars, args=(n, mu),
                       jac=obj_jac, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 800, 'ftol': 1e-12})
        best_vars = res.x
        mu *= 1.5
        
        # Early noise injection to escape local minima
        if step < 10:
            best_vars += np.random.normal(0, 0.0005, size=len(best_vars))
            best_vars = np.clip(best_vars, 0.0, 1.0)
            
    final_centers = best_vars[:2 * n].reshape(n, 2)
    final_radii = best_vars[2 * n:]
    
    # Safety shrink to guarantee strict validity within 1e-12 tolerance
    final_radii *= 0.9999999
    
    return final_centers, final_radii, np.sum(final_radii)
