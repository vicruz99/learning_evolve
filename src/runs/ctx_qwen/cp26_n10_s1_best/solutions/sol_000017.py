# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0cda7bbd) state=2f6d054e sum of radii=2.596717 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    
    for i in range(n):
        if radii[i] < 0:
            return False
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

def get_initial_guess(n, seed):
    """Generate an initial configuration of n circles."""
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05 # Start with small radii
    
    # Strategy 1: Grid-like distribution
    # 5x5 grid has 25 points. We need 26.
    # Let's try to place them somewhat evenly.
    
    # Try a hexagonal packing pattern or just random valid points
    # Random points with rejection sampling to ensure no overlap initially?
    # Or just random and let optimizer fix.
    
    # Let's try a perturbed grid for stability
    # 6 columns, 5 rows = 30 spots. Pick 26.
    grid_x = np.linspace(0.1, 0.9, 6)
    grid_y = np.linspace(0.1, 0.9, 5)
    
    points = []
    for y in grid_y:
        for x in grid_x:
            points.append([x, y])
            
    # We have 30 points. Select 26.
    # To make it interesting, maybe random selection or specific pattern.
    # Let's just take the first 26.
    selected_points = points[:n]
    
    centers = np.array(selected_points)
    # Add some noise to break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    # Clip to valid range for centers (roughly)
    centers = np.clip(centers, 0.1, 0.9)
    
    return centers, radii

def objective_function(vars, n):
    """Objective: maximize sum of radii -> minimize negative sum."""
    radii = vars[2::3]
    return -np.sum(radii)

def constraint_bound(vars, n):
    """Boundary constraints: r <= x <= 1-r, r <= y <= 1-r"""
    c = []
    for i in range(n):
        x = vars[3*i]
        y = vars[3*i+1]
        r = vars[3*i+2]
        # x - r >= 0
        c.append(x - r)
        # 1 - x - r >= 0
        c.append(1 - x - r)
        # y - r >= 0
        c.append(y - r)
        # 1 - y - r >= 0
        c.append(1 - y - r)
        # r >= 0
        c.append(r)
    return np.array(c)

def constraint_overlap(vars, n):
    """Non-overlap constraints: dist^2 >= (r1+r2)^2"""
    c = []
    for i in range(n):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist_sq = (ri + rj)**2
            c.append(dist_sq - min_dist_sq)
    return np.array(c)

def solve_packing(n=26, seed=42):
    centers_init, radii_init = get_initial_guess(n, seed)
    
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = []
    for i in range(n):
        x0.extend([centers_init[i, 0], centers_init[i, 1], radii_init[i]])
    
    # Bounds for variables
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Constraints
    # We need to pass constraints to minimize.
    # SLSQP can handle a single constraint function returning a vector?
    # Actually, it's safer to pass a list of NonlinearConstraint or dicts.
    # But with many constraints, it's slow.
    # Let's try to pass them as a single function if possible?
    # No, standard interface is list.
    
    # Optimization with SLSQP
    # To speed up, maybe we can use a penalty method?
    # But let's try standard first.
    
    # Define constraints as dicts
    cons = []
    
    # Boundary constraints can be split into individual or vector?
    # Let's create a function that returns all boundary constraints
    cons.append({
        'type': 'ineq',
        'fun': lambda vars: constraint_bound(vars, n)
    })
    
    # Overlap constraints
    cons.append({
        'type': 'ineq',
        'fun': lambda vars: constraint_overlap(vars, n)
    })

    try:
        res = minimize(
            objective_function,
            x0,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if res.success or res.fun > -10: # Check if reasonable
            best_centers = np.zeros((n, 2))
            best_radii = np.zeros(n)
            for i in range(n):
                best_centers[i, 0] = res.x[3*i]
                best_centers[i, 1] = res.x[3*i+1]
                best_radii[i] = res.x[3*i+2]
            
            # Validate and slightly shrink if needed to fix numerical errors
            if validate_packing(best_centers, best_radii):
                return best_centers, best_radii, np.sum(best_radii)
            else:
                # Try to fix by shrinking radii slightly
                # Or just return invalid? No, must be valid.
                # Let's scale down radii slightly until valid
                factor = 1.0
                for _ in range(100):
                    test_radii = best_radii * factor
                    if validate_packing(best_centers, test_radii):
                        best_radii = test_radii
                        return best_centers, best_radii, np.sum(best_radii)
                    factor *= 0.99
                return best_centers, best_radii * 0.9, 0.0 # Fallback
        else:
            return None, None, 0.0
            
    except Exception as e:
        return None, None, 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple seeds/configurations
    seeds = [42, 123, 456, 789, 0, 1, 2, 3, 4, 5, 10, 20, 50, 100]
    
    # Also try specific structured initial guesses
    # 1. Random grid
    # 2. Hexagonal
    # 3. Concentric?
    
    for seed in seeds:
        centers, radii, s = solve_packing(26, seed)
        if centers is not None and s > best_sum:
            best_sum = s
            best_centers = centers
            best_radii = radii
            
        # Try to improve the best found so far by using its result as seed
        if best_centers is not None:
             # Perturb and re-optimize
             # We can just pass the current best as initial guess?
             # But solve_packing generates its own init.
             # Let's implement a refine step.
             pass

    # If we have a good solution, try to refine it further
    if best_centers is not None:
        # Use the best centers as initial guess for a new optimization
        # We need to construct x0 manually
        x0 = []
        for i in range(26):
            x0.extend([best_centers[i, 0], best_centers[i, 1], best_radii[i]])
        
        n = 26
        bounds = []
        for _ in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])
            
        cons = [
            {'type': 'ineq', 'fun': lambda vars: constraint_bound(vars, n)},
            {'type': 'ineq', 'fun': lambda vars: constraint_overlap(vars, n)}
        ]
        
        try:
            res = minimize(
                objective_function,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-10}
            )
            
            if res.success:
                new_centers = np.zeros((n, 2))
                new_radii = np.zeros(n)
                for i in range(n):
                    new_centers[i, 0] = res.x[3*i]
                    new_centers[i, 1] = res.x[3*i+1]
                    new_radii[i] = res.x[3*i+2]
                
                if validate_packing(new_centers, new_radii):
                    new_sum = np.sum(new_radii)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_centers = new_centers
                        best_radii = new_radii
                else:
                    # Try to fix
                    factor = 1.0
                    for _ in range(100):
                        if validate_packing(new_centers, new_radii * factor):
                            best_sum = np.sum(new_radii * factor)
                            best_centers = new_centers
                            best_radii = new_radii * factor
                            break
                        factor *= 0.99
        except Exception:
            pass

    if best_centers is None:
        # Fallback to a simple valid packing
        # 5x5 grid of radius 0.1 (25 circles) + 1 small circle?
        # But we need 26.
        # 26 circles of radius 0.09?
        # 26 * 0.09 = 2.34.
        # Let's generate a valid fallback.
        best_centers = np.zeros((26, 2))
        best_radii = np.ones(26) * 0.08
        # Place in grid
        idx = 0
        for r_idx in range(5):
            for c_idx in range(5):
                if idx < 25:
                    best_centers[idx, 0] = 0.1 + c_idx * 0.2
                    best_centers[idx, 1] = 0.1 + r_idx * 0.2
                    idx += 1
        if idx < 26:
            best_centers[25, 0] = 0.5
            best_centers[25, 1] = 0.5
            best_radii[25] = 0.01 # tiny
        best_sum = np.sum(best_radii)
        # Validate fallback
        if not validate_packing(best_centers, best_radii):
             # Just ensure valid
             best_radii = np.ones(26) * 0.05
             best_sum = 26 * 0.05

    return best_centers, best_radii, best_sum
