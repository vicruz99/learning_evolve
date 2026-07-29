# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8a979775) state=1cb5ec92 sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize


def compute_initial_hexagonal(n):
    """Create initial positions using hexagonal packing pattern."""
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.085)
    
    idx = 0
    y = 0.1
    row = 0
    x_base = 0.12
    
    while idx < n:
        if row % 2 == 0:
            x = x_base
            x_spacing = 0.175
        else:
            x = x_base + 0.0875
            x_spacing = 0.175
        
        while x <= 1.0 - 0.1 and idx < n:
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            x += x_spacing
        
        y += 0.14
        row += 1
    
    return centers[:n], radii[:n]


def obj_func(x, n):
    """Objective function: negative sum of radii (to maximize)."""
    return -np.sum(x[2 * n:])


def constraint_func(x, n):
    """Compute all constraint values (must be >= 0)."""
    centers = x[:2 * n].reshape((n, 2))
    radii = x[2 * n:]
    
    constraints = []
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    for i in range(n):
        constraints.append(centers[i, 0] - radii[i])
        constraints.append(1.0 - centers[i, 0] - radii[i])
        constraints.append(centers[i, 1] - radii[i])
        constraints.append(1.0 - centers[i, 1] - radii[i])
    
    # Non-overlap constraints: distance >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            constraints.append(dist - radii[i] - radii[j])
    
    # Non-negative radii
    for i in range(n):
        constraints.append(radii[i])
    
    return np.array(constraints)


def refine_packing(centers, radii, n):
    """Refine packing using iterative local optimization."""
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(50):
        improved = False
        for i in range(n):
            # Calculate maximum possible radius for circle i
            max_r = min(
                centers[i, 0],
                1.0 - centers[i, 0],
                centers[i, 1],
                1.0 - centers[i, 1]
            )
            
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - radii[j])
            
            max_r = max(0, max_r)
            
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        
        if not improved:
            break
    
    return centers, radii


def run_packing():
    """Pack 26 circles in unit square to maximize sum of radii."""
    n = 26
    
    # Create bounds for optimization
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': lambda x: constraint_func(x, n)}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Multiple restarts from different initial configurations
    for trial in range(15):
        if trial == 0:
            # Hexagonal initialization
            centers_init, radii_init = compute_initial_hexagonal(n)
        elif trial == 1:
            # Grid initialization
            centers_init = np.zeros((n, 2))
            idx = 0
            for row in range(5):
                for col in range(6):
                    if idx >= n:
                        break
                    centers_init[idx, 0] = 0.1 + col * 0.18
                    centers_init[idx, 1] = 0.1 + row * 0.18
                    idx += 1
            radii_init = np.full(n, 0.08)
        elif trial == 2:
            # Corner-focused initialization
            centers_init = np.zeros((n, 2))
            corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            idx = 0
            for cx, cy in corners:
                centers_init[idx] = [cx, cy]
                idx += 1
            # Fill rest in hexagonal pattern
            remaining = compute_initial_hexagonal(n - 4)
            centers_init[idx:idx+remaining[0].shape[0]] = remaining[0]
            radii_init = np.full(n, 0.085)
        else:
            # Perturbed versions of best solution found so far
            if best_centers is not None:
                centers_init = best_centers.copy()
                radii_init = best_radii.copy()
                centers_init += np.random.normal(0, 0.005, centers_init.shape)
                centers_init = np.clip(centers_init, 0.02, 0.98)
            else:
                centers_init, radii_init = compute_initial_hexagonal(n)
                centers_init += np.random.normal(0, 0.02, centers_init.shape)
                centers_init = np.clip(centers_init, 0.05, 0.95)
        
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        # Run optimization
        try:
            result = minimize(
                obj_func,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )
            
            current_sum = -result.fun
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = result.x[:2 * n].reshape((n, 2))
                best_radii = result.x[2 * n:].copy()
        except Exception:
            continue
    
    if best_centers is None:
        best_centers, best_radii = compute_initial_hexagonal(n)
        best_sum = np.sum(best_radii)
    
    # Ensure radii are non-negative
    best_radii = np.maximum(best_radii, 0)
    
    # Refinement pass
    best_centers, best_radii = refine_packing(best_centers, best_radii, n)
    
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum
