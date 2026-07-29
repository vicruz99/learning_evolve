# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96e346d6) state=f3800353 sum of radii=2.541000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    n = 26
    np.random.seed(42) # For reproducibility

    # Helper to generate initial hexagonal positions
    def generate_hex_grid(n, r):
        centers = []
        # Try to fit circles in rows
        row_y = r
        row_idx = 0
        while len(centers) < n:
            # Determine x positions for this row
            # Shift every other row by r (half diameter)
            shift = r if row_idx % 2 == 1 else 0
            x = r + shift
            while x <= 1 - r:
                centers.append((x, row_y))
                x += 2 * r
            row_y += r * math.sqrt(3)
            row_idx += 1
        
        # If we have fewer than n, add random ones in gaps or just extend
        while len(centers) < n:
            centers.append((np.random.rand(), np.random.rand()))
        
        # Return first n
        return np.array(centers[:n])

    # Initial guess for radius. 
    # 25 circles fit at r=0.1. 26 is slightly tighter.
    # Hexagonal packing is denser.
    # Estimate r based on area: 26 * pi * r^2 approx 0.9 * 1 => r approx 0.105
    # But boundary constraints reduce this. Let's start with 0.09 and grow.
    r_start = 0.095
    
    # Generate initial centers
    # Try to pack them in a hexagonal pattern
    centers = generate_hex_grid(n, r_start)
    
    # Ensure centers are within bounds [r, 1-r] roughly
    # Clip to valid range for current r
    centers = np.clip(centers, r_start, 1 - r_start)

    # Optimization function to maximize radius r
    # We will treat r as a variable and centers as variables.
    # However, maximizing r with constraints is non-convex.
    # We can use a penalty method or just optimize positions for a fixed r and binary search r.
    # Given time constraints, a single optimization run with penalty is safer.
    
    # Variables: [x1, y1, ..., x26, y26, r]
    # Size: 52 + 1 = 53
    
    initial_vars = np.hstack([centers.flatten(), [r_start]])
    
    def objective(vars):
        # We want to maximize r, so minimize -r
        # But we need to penalize constraint violations heavily
        centers_opt = vars[:-1].reshape((n, 2))
        r_opt = vars[-1]
        
        penalty = 0.0
        weight = 1000.0 # High weight for constraints
        
        # Boundary constraints
        # x >= r, x <= 1-r => r <= x <= 1-r
        # Violation: if x < r, penalty (r-x)^2. If x > 1-r, penalty (x-(1-r))^2
        for i in range(n):
            x, y = centers_opt[i]
            # Check x
            if x < r_opt:
                penalty += weight * (x - r_opt)**2
            elif x > 1 - r_opt:
                penalty += weight * (x - (1 - r_opt))**2
            
            # Check y
            if y < r_opt:
                penalty += weight * (y - r_opt)**2
            elif y > 1 - r_opt:
                penalty += weight * (y - (1 - r_opt))**2
        
        # Overlap constraints
        # dist >= 2r => dist^2 >= 4r^2
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = 4 * r_opt**2
                
                if dist_sq < min_dist_sq:
                    # Penalty proportional to penetration depth squared?
                    # Or just (min_dist - dist)^2
                    dist = math.sqrt(dist_sq)
                    penetration = min_dist_sq - dist_sq # Not exactly depth, but monotonic
                    # Better: (2r - dist)^2
                    depth = 2 * r_opt - dist
                    if depth > 0:
                        penalty += weight * depth**2
                        
        return -r_opt + penalty

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0.01, 0.5)]
    
    # Use SLSQP or similar. SLSQP is good for constrained but we used penalty.
    # Actually, let's use 'L-BFGS-B' with bounds, since constraints are in objective.
    # Or 'SLSQP' if we add constraints explicitly, but penalty is easier for vectorization.
    
    # To improve results, we can run multiple restarts or a sequence of refinements.
    # Let's try a single run first with a good initialization.
    
    # Refine initialization: Place circles in a slightly tighter grid if possible
    # Or random restart.
    
    best_result = None
    best_score = -np.inf
    
    # Try a few random restarts to escape local minima
    for restart in range(5):
        # Random initialization
        np.random.seed(42 + restart)
        c_init = np.random.rand(n, 2)
        # Clip to allow some margin
        c_init = np.clip(c_init, 0.05, 0.95)
        r_init = 0.08 # Start small
        vars_init = np.hstack([c_init.flatten(), [r_init]])
        
        res = minimize(objective, vars_init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success or res.nit > 100:
            val = res.fun
            # Extract r
            r_final = res.x[-1]
            # Check actual penalty (if penalty is 0, constraint satisfied)
            # The objective is -r + penalty.
            # If penalty is small, r is good.
            # We want to maximize r, so minimize -r.
            # But penalty adds positive value.
            # So we want low objective value? 
            # Wait, objective = -r + penalty.
            # If valid, penalty ~ 0, obj ~ -r.
            # We want obj to be as negative as possible (large r).
            # However, if penalty is high, obj is positive.
            # We should check the validity of the solution found.
            
            centers_found = res.x[:-1].reshape((n, 2))
            
            # Validate and calculate penalty manually
            p = 0
            valid = True
            for i in range(n):
                x, y = centers_found[i]
                if x < r_final - 1e-6 or x > 1 - r_final + 1e-6: valid = False
                if y < r_final - 1e-6 or y > 1 - r_final + 1e-6: valid = False
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(centers_found[i] - centers_found[j])
                        if d < 2 * r_final - 1e-6:
                            valid = False
                            break
                if valid:
                    # Valid packing found
                    current_sum = 26 * r_final
                    if current_sum > best_score:
                        best_score = current_sum
                        best_result = (centers_found.copy(), np.full(n, r_final))
    
    # If we didn't find a valid one with high r, fallback to a known safe grid or result
    if best_result is None:
        # Fallback: 5x5 grid + 1 small circle?
        # Or just the best attempt even if slightly invalid? 
        # Validation function is strict. We must return valid.
        # Let's construct a valid 26 circle packing manually as fallback.
        # 25 circles of radius 0.1 in grid.
        # 1 circle in center gap?
        # Grid centers: 0.1, 0.3, 0.5, 0.7, 0.9
        centers_fb = []
        for i in range(5):
            for j in range(5):
                centers_fb.append([0.1 + i*0.2, 0.1 + j*0.2])
        # 26th circle?
        # Center of square is (0.5, 0.5). Occupied by circle 12 (index).
        # Gap between (0.3, 0.3), (0.3, 0.5), (0.5, 0.3), (0.5, 0.5)?
        # Gap center (0.4, 0.4).
        # Distance to (0.3, 0.3) is sqrt(0.01+0.01) = 0.1414.
        # Radius 0.1 circle occupies up to 0.1.
        # Gap radius = 0.1414 - 0.1 = 0.0414.
        centers_fb.append([0.4, 0.4])
        radii_fb = np.array([0.1]*25 + [0.041]) # Slightly less to be safe
        best_result = (np.array(centers_fb), radii_fb)

    centers_sol, radii_sol = best_result
    sum_r = np.sum(radii_sol)
    
    # Final validation check before returning
    # Just to be safe
    if not validate_packing(centers_sol, radii_sol):
        # If failed, return the fallback explicitly
        centers_fb = []
        for i in range(5):
            for j in range(5):
                centers_fb.append([0.1 + i*0.2, 0.1 + j*0.2])
        centers_fb.append([0.4, 0.4])
        radii_fb = np.array([0.1]*25 + [0.04])
        centers_sol = np.array(centers_fb)
        radii_sol = radii_fb
        sum_r = np.sum(radii_sol)

    return centers_sol, radii_sol, sum_r

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    import numpy as np
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

# Note: The prompt requires run_packing to be defined. 
# The validate_packing function is provided in the prompt description, 
# but I included it here for completeness if needed, though the user's validator will be used.
# I will rely on the logic inside run_packing.
