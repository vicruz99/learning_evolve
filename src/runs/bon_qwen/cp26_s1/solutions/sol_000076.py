# sol_000076 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22281c24) state=2cccf46a sum of radii=1.300000 correctness=1.0
# stdout(first 200): Circle 2 at (0.5000000000004273, 0.5000000000004319) with radius 0.5000000000005805 is outside the unit square Circles 0 and 1 overlap: dist=2.7307875685720004e-15, r1+r2=0.9999999999999918 Circles 0 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

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

def compute_distance(x):
    """
    Computes the sum of radii (negated for minimization) and constraints.
    Actually, we will separate objective and constraints for clarity in the optimizer.
    """
    pass

def objective(vars, n):
    """
    Objective function: Minimize negative sum of radii.
    vars layout: [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = vars[2::3]
    return -np.sum(radii)

def constraints_factory(n):
    """
    Factory to create constraints for the optimizer.
    Returns a list of constraint dictionaries.
    """
    constraints = []
    
    # Boundary constraints
    # For each circle i:
    # x_i - r_i >= 0  =>  x_i - r_i >= 0
    # x_i + r_i <= 1  =>  1 - (x_i + r_i) >= 0
    # y_i - r_i >= 0
    # y_i + r_i <= 1
    # r_i >= 0 (handled by bounds usually, but good to be safe)
    
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x - r >= 0
        def constr_xr_min(v, i=i):
            return v[3*i] - v[3*i+2]
        constraints.append({'type': 'ineq', 'fun': constr_xr_min})
        
        # 1 - (x + r) >= 0
        def constr_xr_max(v, i=i):
            return 1.0 - (v[3*i] + v[3*i+2])
        constraints.append({'type': 'ineq', 'fun': constr_xr_max})
        
        # y - r >= 0
        def constr_yr_min(v, i=i):
            return v[3*i+1] - v[3*i+2]
        constraints.append({'type': 'ineq', 'fun': constr_yr_min})
        
        # 1 - (y + r) >= 0
        def constr_yr_max(v, i=i):
            return 1.0 - (v[3*i+1] + v[3*i+2])
        constraints.append({'type': 'ineq', 'fun': constr_yr_max})
        
        # r >= 0 (epsilon to ensure positive)
        def constr_r_pos(v, i=i):
            return v[3*i+2] - 1e-6
        constraints.append({'type': 'ineq', 'fun': constr_r_pos})

    # Overlap constraints
    # dist(i, j) >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 3 * i
            idx_yi = 3 * i + 1
            idx_ri = 3 * i + 2
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            def constr_overlap(v, i=i, j=j):
                dx = v[idx_xi] - v[idx_xj]
                dy = v[idx_yi] - v[idx_yj]
                dr = v[idx_ri] + v[idx_rj]
                return dx*dx + dy*dy - dr*dr
            
            constraints.append({'type': 'ineq', 'fun': constr_overlap})
            
    return constraints

def generate_initial_guess(n, seed_offset=0):
    """
    Generates an initial guess for circle centers and radii.
    Uses a hexagonal-like packing perturbed slightly.
    """
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09) # Start with small valid radius
    
    # Attempt a hexagonal packing layout
    # 5 rows, varying number of circles
    # Row 0: 5, Row 1: 6, Row 2: 5, Row 3: 6, Row 4: 4 -> Total 26
    # Let's try to distribute 26 circles roughly evenly
    
    rows = 6
    cols = 5 # Approx
    
    idx = 0
    # We want to fit n circles.
    # Let's try a grid approach first then perturb
    # 5x5 grid is 25. We need 1 more.
    # Let's place 25 in 5x5 and 1 in center? No, center is occupied.
    # Let's place 26 in a slightly distorted grid.
    
    # Better heuristic: Random uniform with small radius, then let optimizer work?
    # Or a dense packing.
    # Let's try a 6x5 grid (30 spots) and pick 26?
    # Or just a randomized perturbation of a dense grid.
    
    # Let's construct a hexagonal pattern for 26 circles.
    # Approximate positions
    # Row height = sqrt(3)/2 * 2r = r*sqrt(3). With r=0.1, height ~ 0.1732
    # Horizontal spacing = 2r = 0.2
    
    # Let's fit into [0,1]x[0,1]
    # Try 5 rows.
    # Row y coords: 0.1, 0.1+0.1732, 0.1+0.3464, 0.1+0.5196, 0.1+0.6928
    # 0.1, 0.2732, 0.4464, 0.6196, 0.7928. (5 rows fit).
    
    y_coords = [0.12, 0.3, 0.48, 0.66, 0.84, 0.05] # 6 rows to spread out?
    # Let's just use a deterministic pattern that fills the space
    
    # Strategy: Fill rows with 5 or 6 circles.
    # 26 = 6 + 5 + 6 + 5 + 4 ?
    # 6 circles in a row needs width ~ 1.2 if r=0.1. Too wide.
    # So rows must have <= 5 circles if r ~ 0.1.
    # If rows have 5 circles, we need 6 rows for 30 circles.
    # We need 26. So 5 rows of 5 (25) + 1 extra.
    # Where to put the 26th?
    # Maybe in a gap.
    
    # Let's use a randomized perturbation of a 5x5 grid + 1 center.
    # But 5x5 grid centers are at 0.1, 0.3, 0.5, 0.7, 0.9.
    # Gap centers at 0.2, 0.4, 0.6, 0.8.
    
    # Let's generate 26 points.
    points = []
    # 5x5 grid
    for r in range(5):
        for c in range(5):
            points.append([0.1 + 0.2 * c, 0.1 + 0.2 * r])
    # Add 26th point in a gap, e.g., (0.2, 0.2) or (0.5, 0.2)
    # (0.5, 0.2) is equidistant from (0.3, 0.1), (0.7, 0.1), (0.5, 0.3) ?
    # Let's add it at (0.2, 0.2) which is a gap corner.
    points.append([0.2, 0.2])
    
    # This leaves the 25 grid circles at 0.1 radius.
    # The 26th circle at (0.2, 0.2) would overlap with (0.1, 0.1) if r=0.1.
    # Distance from (0.1, 0.1) to (0.2, 0.2) is sqrt(0.02) ~ 0.141.
    # Sum of radii 0.2. Overlap.
    # So we must shrink radii or move centers.
    # Optimizer will handle this.
    
    # Let's shuffle points to avoid ordering bias
    np.random.seed(42 + seed_offset)
    np.random.shuffle(points)
    
    centers = np.array(points)
    # Initialize radii small to avoid immediate violation
    radii = np.full(n, 0.05)
    
    return centers, radii

def flatten(centers, radii):
    """Flatten centers and radii into a 1D array."""
    n = centers.shape[0]
    vars = np.zeros(3 * n)
    for i in range(n):
        vars[3*i] = centers[i, 0]
        vars[3*i+1] = centers[i, 1]
        vars[3*i+2] = radii[i]
    return vars

def unflatten(vars, n):
    """Unflatten 1D array back to centers and radii."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = vars[3*i]
        centers[i, 1] = vars[3*i+1]
        radii[i] = vars[3*i+2]
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    constraints = constraints_factory(n)
    
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple restarts with different initializations
    for seed in range(5):
        centers, radii = generate_initial_guess(n, seed)
        x0 = flatten(centers, radii)
        
        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        try:
            # Use SLSQP for constrained optimization
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -best_sum_radii + 0.001: # Allow some tolerance
                opt_centers, opt_radii = unflatten(res.x, n)
                current_sum = np.sum(opt_radii)
                
                # Validate the result just in case
                if validate_packing(opt_centers, opt_radii):
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = opt_centers.copy()
                        best_radii = opt_radii.copy()
                else:
                    # If validation fails, try to repair or discard
                    # Sometimes numerical error causes tiny violations.
                    # We can try to clamp radii slightly or re-optimize.
                    # For now, let's just keep it if it's close.
                    # But strict validation is required for final output.
                    # Let's trust the optimizer but be careful.
                    pass
                    
        except Exception:
            continue
            
    # Fallback if optimization failed or didn't improve
    if best_centers is None:
        centers, radii = generate_initial_guess(n, 0)
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)
        
    # Final validation check
    # If validation fails, we might need to adjust.
    # However, with SLSQP and tight constraints, it should be valid.
    # But due to float precision, might be slightly off.
    # The validation function has a tolerance of 1e-12.
    
    return best_centers, best_radii, float(best_sum_radii)

# Note: The validate_packing function is provided in the prompt and should not be modified.
# I included it above for context but the final code block should only contain the solution.
# The prompt says "You must define the run_packing function".
# It also says "Make all helper functions top level".
# I will include the validation function in the final output just to be safe?
# "We will run the below validation function (read-only, do not modify this)"
# Usually this means the environment has it. I don't need to include it in my code unless requested.
# But to be self-contained, I won't include it unless necessary. 
# The prompt asks to return the program. I will include my helper functions and run_packing.
