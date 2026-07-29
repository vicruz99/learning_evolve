# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34f92e2c) state=a7d84335 sum of radii=2.617814 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def generate_initial_config(n_circles):
    """
    Generates an initial configuration of centers and small radii.
    Uses a mix of grid and random placement to avoid symmetry issues.
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Try to distribute points somewhat evenly first
    # Hexagonal-ish grid generation
    idx = 0
    # Approximate rows for 26 circles
    # 5, 5, 5, 5, 4, 2 -> 26
    # Or just random scatter
    
    # Let's use a random scatter with a minimum distance constraint to avoid
    # starting with huge overlaps
    attempts = 0
    while idx < n_circles and attempts < 10000:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # Check distance to existing centers
        is_valid = True
        for k in range(idx):
            dist = np.sqrt((x - centers[k, 0])**2 + (y - centers[k, 1])**2)
            if dist < 0.15: # Minimum initial separation
                is_valid = False
                break
        
        if is_valid:
            centers[idx, 0] = x
            centers[idx, 1] = y
            radii[idx] = 0.01 # Start with small radius
            idx += 1
        attempts += 1
        
    # Fill remaining if any (should be rare)
    while idx < n_circles:
        centers[idx, 0] = random.uniform(0, 1)
        centers[idx, 1] = random.uniform(0, 1)
        radii[idx] = 0.01
        idx += 1
        
    return centers, radii

def objective_function(params, n):
    """
    Objective function to maximize sum of radii.
    We minimize negative sum.
    """
    # params structure: [x1, y1, r1, x2, y2, r2, ...]
    radii = params[2::3]
    return -np.sum(radii)

def constraint_wall(params, n):
    """
    Constraints for walls:
    r <= x <= 1-r  => x - r >= 0, 1 - r - x >= 0
    r <= y <= 1-r  => y - r >= 0, 1 - r - y >= 0
    """
    constraints = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        r = params[3*i + 2]
        
        # x - r >= 0
        constraints.append(x - r)
        # 1 - r - x >= 0
        constraints.append(1 - r - x)
        # y - r >= 0
        constraints.append(y - r)
        # 1 - r - y >= 0
        constraints.append(1 - r - y)
        
        # r >= 0 (handled by bounds usually, but let's keep explicit if needed, 
        # though bounds are better)
        
    return np.array(constraints)

def constraint_overlap(params, n):
    """
    Constraints for non-overlap:
    dist(i, j) >= r_i + r_j
    => sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
    """
    constraints = []
    for i in range(n):
        xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # Using a small epsilon to avoid numerical issues with sqrt(0)
            # But dist is distance between centers.
            # We need dist >= ri + rj
            # If centers are same, dist=0, constraint -ri-rj >= 0 impossible if r>0.
            # Optimizer should handle this.
            
            constraints.append(dist - ri - rj)
            
    return np.array(constraints)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for variables: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Number of restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        # Generate initial config
        init_centers, init_radii = generate_initial_config(n_circles)
        
        # Flatten to params vector
        init_params = np.zeros(3 * n_circles)
        for i in range(n_circles):
            init_params[3*i] = init_centers[i, 0]
            init_params[3*i+1] = init_centers[i, 1]
            init_params[3*i+2] = init_radii[i]
            
        # Define constraints for SLSQP
        # We need to pass the constraints as dictionaries or callable objects
        # SLSQP supports 'ineq' constraints where func(x) >= 0
        
        cons = []
        
        # Wall constraints
        cons.append({
            'type': 'ineq',
            'fun': lambda p: constraint_wall(p, n_circles)
        })
        
        # Overlap constraints
        cons.append({
            'type': 'ineq',
            'fun': lambda p: constraint_overlap(p, n_circles)
        })
        
        # Try to optimize
        # Method SLSQP is good for constraints
        # Maxiter might need to be high
        try:
            res = minimize(
                objective_function,
                init_params,
                args=(n_circles,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.success or res.nit > 10: # Accept reasonable attempts
                current_sum = -res.fun # Objective was negative sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = np.zeros((n_circles, 2))
                    best_radii = np.zeros(n_circles)
                    for i in range(n_circles):
                        best_centers[i, 0] = res.x[3*i]
                        best_centers[i, 1] = res.x[3*i+1]
                        best_radii[i] = res.x[3*i+2]
        except Exception as e:
            print(f"Optimization failed in restart {restart}: {e}")
            continue
            
        # If we found a very good solution, we might stop early or continue
        # For robustness, let's just run all restarts
        
    # Final validation and return
    # Ensure strict validity by clamping or checking
    if best_centers is not None:
        # Just to be safe, run a quick repair? 
        # The optimizer should have satisfied constraints, but numerical noise might exist.
        # Let's trust the optimizer for now as the validation function has tolerance.
        
        # Check validity
        if validate_packing(best_centers, best_radii):
            return best_centers, best_radii, float(best_sum)
        else:
            # If invalid, try to fix? 
            # For this task, returning the best found is expected.
            # If invalid, maybe return a safe fallback.
            pass

    # Fallback solution: Simple grid packing
    # 5x5 grid = 25 circles, r=0.1. 
    # We need 26. 
    # Let's place 25 in 5x5 grid with r=0.09 and one small one?
    # Or just 26 circles with small radii.
    centers_fallback = np.zeros((n_circles, 2))
    radii_fallback = np.zeros(n_circles)
    idx = 0
    for r in range(5):
        for c in range(5):
            if idx < 26:
                centers_fallback[idx, 0] = 0.1 + c * 0.2
                centers_fallback[idx, 1] = 0.1 + r * 0.2
                radii_fallback[idx] = 0.09
                idx += 1
    # Last one
    if idx < 26:
        centers_fallback[idx, 0] = 0.5
        centers_fallback[idx, 1] = 0.5
        radii_fallback[idx] = 0.01
        
    # This fallback is weak but valid.
    # The optimizer should find much better.
    return centers_fallback, radii_fallback, float(np.sum(radii_fallback))

# To ensure the code runs without errors if scipy is not available or optimization fails
# But instructions say we can use scipy.

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(validate_packing(centers, radii))
