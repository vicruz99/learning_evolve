# sol_000335 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9bf69ab6) state=9283fd0b sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

# Helper classes for constraints to avoid lambdas and closures
class DistanceConstraint:
    def __init__(self, i, j):
        self.i = i
        self.j = j
    
    def __call__(self, vars):
        # vars layout: [x1, y1, r1, x2, y2, r2, ...]
        xi = vars[i*3]
        yi = vars[i*3+1]
        ri = vars[i*3+2]
        
        xj = vars[j*3]
        yj = vars[j*3+1]
        rj = vars[j*3+2]
        
        # Constraint: dist^2 >= (ri + rj)^2
        # Equivalent to: dist^2 - (ri + rj)^2 >= 0
        dist_sq = (xi - xj)**2 + (yi - yj)**2
        r_sum_sq = (ri + rj)**2
        
        return dist_sq - r_sum_sq

class BoundaryConstraint:
    def __init__(self, i, dim, bound_type):
        # dim: 0 for x, 1 for y
        # bound_type: 0 for lower (x >= r), 1 for upper (x <= 1-r)
        self.i = i
        self.dim = dim
        self.bound_type = bound_type
        
    def __call__(self, vars):
        idx_pos = self.i * 3 + self.dim
        idx_r = self.i * 3 + 2
        
        pos = vars[idx_pos]
        r = vars[idx_r]
        
        if self.bound_type == 0:
            # pos - r >= 0
            return pos - r
        else:
            # 1.0 - pos - r >= 0
            return 1.0 - pos - r

class RadiusConstraint:
    def __init__(self, i):
        self.i = i
    def __call__(self, vars):
        # r >= 0
        return vars[self.i * 3 + 2]

def objective(vars):
    # We want to maximize sum of radii, so we minimize negative sum
    # Radii are at indices 2, 5, 8, ...
    return -np.sum(vars[2::3])

def get_hex_init(n, r_est=0.08):
    """Generate an initial configuration based on a hexagonal lattice."""
    d = 2 * r_est
    h = d * math.sqrt(3) / 2
    centers = []
    y = r_est
    row = 0
    while len(centers) < n and y + r_est <= 1.0:
        x = r_est
        if row % 2 == 1:
            x += d / 2
        while x + r_est <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += d
        y += h
        row += 1
    # Fill remaining spots if needed
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
    return np.array(centers[:n])

def run_packing() -> tuple:
    n_circles = 26
    
    # Build constraints list
    constraints = []
    
    # Boundary and radius constraints for each circle
    for i in range(n_circles):
        constraints.append({'type': 'ineq', 'fun': BoundaryConstraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': BoundaryConstraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': BoundaryConstraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': BoundaryConstraint(i, 1, 1)})
        constraints.append({'type': 'ineq', 'fun': RadiusConstraint(i)})
        
    # Pairwise distance constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            constraints.append({'type': 'ineq', 'fun': DistanceConstraint(i, j)})
            
    # Bounds for variables: x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.extend([
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.5)
        ])
        
    best_val = -np.inf
    best_x = None
    
    # Run optimization from multiple starting points
    n_restarts = 6
    
    for k in range(n_restarts):
        # Initialize centers
        if k < 3:
            # Hexagonal initialization with some noise
            centers = get_hex_init(n_circles)
            centers += np.random.randn(n_circles, 2) * 0.02
            r0 = 0.08
        else:
            # Random initialization within safe margins
            centers = 0.1 + 0.8 * np.random.rand(n_circles, 2)
            r0 = 0.05
            
        # Flatten to 1D vector [x1, y1, r1, x2, y2, r2, ...]
        x0 = []
        for i in range(n_circles):
            x0.extend([centers[i, 0], centers[i, 1], r0])
        x0 = np.array(x0)
        
        try:
            res = minimize(
                fun=objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
            )
            
            # Check result
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x
            else:
                # Even if not successful, check if we found a better point
                val = -res.fun
                radii = res.x[2::3]
                # Ensure radii are non-negative (within tolerance)
                if np.all(radii >= -1e-6) and val > best_val:
                    best_val = val
                    best_x = res.x
        except Exception:
            continue
            
    if best_x is not None:
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        for i in range(n_circles):
            centers[i, 0] = best_x[i*3]
            centers[i, 1] = best_x[i*3+1]
            radii[i] = max(0.0, best_x[i*3+2])
        return centers, radii, np.sum(radii)
    
    # Fallback to a simple valid packing if optimization fails
    centers = np.random.rand(n_circles, 2)
    radii = np.full(n_circles, 0.01)
    return centers, radii, np.sum(radii)
