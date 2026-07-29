# sol_000287 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5952a474) state=17b37a08 sum of radii=2.610153 correctness=1.0
# stdout(first 200): Circle 6 at (0.11591646034185463, 0.2598508871028769) with radius 0.11591646034333258 is outside the unit square
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0

    def objective(vars):
        # Objective: minimize negative sum of radii (maximize sum of radii)
        # vars structure: [x1, y1, r1, x2, y2, r2, ...]
        radii = vars[2::3]
        return -np.sum(radii)

    def boundary_constraints(vars):
        cons = []
        for i in range(n_circles):
            x = vars[3 * i]
            y = vars[3 * i + 1]
            r = vars[3 * i + 2]
            # x - r >= 0  => r - x <= 0 (inequality constraint >= 0 is standard for scipy, so x - r >= 0)
            # x + r <= 1  => 1 - (x + r) >= 0
            # y - r >= 0
            # y + r <= 1
            # r >= 0
            
            # scipy constraints: fun(x) >= 0
            cons.append(x - r)
            cons.append(1.0 - (x + r))
            cons.append(y - r)
            cons.append(1.0 - (y + r))
            cons.append(r) # r >= 0
        return cons

    def overlap_constraints(vars):
        cons = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                # dist >= r_i + r_j
                # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                # To avoid sqrt issues with gradients, we can use dist_sq >= (ri+rj)^2
                # But strictly, distance constraint is non-convex.
                # Let's use the distance form for clarity, scipy handles it.
                dist = np.sqrt(max(0, dist_sq))
                cons.append(dist - (ri + rj))
        return cons

    def create_initial_config(strategy='grid'):
        vars = np.zeros(n_circles * 3)
        if strategy == 'grid':
            # Hexagonal grid initialization
            rows = 5
            cols = 6 # Max width
            
            # We need 26 circles. 
            # Pattern: 6, 5, 6, 5, 4 -> 26
            counts = [6, 5, 6, 5, 4]
            
            # Approximate spacing to fit in 1x1
            # Width: 6 circles -> 12r <= 1 => r <= 0.083. 
            # Let's start with r=0.05
            r_start = 0.05
            spacing_x = 2 * r_start * 1.5 # slightly larger to avoid overlap
            spacing_y = 2 * r_start * np.sqrt(3) / 2 * 1.5
            
            # Center the grid
            # Max width approx 12*r_start. Max height approx 8*r_start.
            # Let's just place them nicely.
            
            idx = 0
            y = 0.15 # Start near bottom
            
            for r_idx, count in enumerate(counts):
                # Calculate x positions for this row
                # Total width for 'count' circles is (count-1)*2r + 2r = count*2r
                # We want to center them.
                row_width = (count - 1) * spacing_x
                start_x = (1.0 - row_width) / 2
                
                for c in range(count):
                    if idx >= n_circles: break
                    
                    x = start_x + c * spacing_x
                    # Shift alternate rows for hexagonal packing
                    if r_idx % 2 == 1:
                        x += spacing_x / 2
                        
                    vars[3 * idx] = x
                    vars[3 * idx + 1] = y
                    vars[3 * idx + 2] = r_start
                    
                    idx += 1
                y += spacing_y
        else:
            # Random initialization
            for i in range(n_circles):
                vars[3*i] = np.random.uniform(0.2, 0.8)
                vars[3*i+1] = np.random.uniform(0.2, 0.8)
                vars[3*i+2] = 0.05
        
        return vars

    # Combine constraints
    # Note: scipy minimize expects a list of constraint dicts or a single function returning array
    # Using list of dicts for mixed types or a function returning array for bounds-like
    # SLSQP can take a function that returns an array of constraint values >= 0
    
    def all_constraints(vars):
        c_bound = boundary_constraints(vars)
        c_overlap = overlap_constraints(vars)
        return np.array(c_bound + c_overlap)

    # Bounds for variables to help optimizer
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    best_run = 0
    
    # Try multiple starts
    for i in range(20):
        if i == 0:
            x0 = create_initial_config('grid')
        else:
            x0 = create_initial_config('random')
            # Perturb grid for variety
            if i < 5:
                base = create_initial_config('grid')
                x0 = base + np.random.normal(0, 0.01, base.shape)
                # Ensure within bounds
                x0[::3] = np.clip(x0[::3], 0, 1)
                x0[1::3] = np.clip(x0[1::3], 0, 1)
                x0[2::3] = np.clip(x0[2::3], 0.01, 0.2)

        # Define constraints for SLSQP
        constraints = {'type': 'ineq', 'fun': all_constraints}

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success or res.fun < -1.5: # If found a decent solution
                current_sum = -res.fun
                if current_sum > best_sum:
                    # Validate the result
                    centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n_circles)])
                    radii = np.array([res.x[3*i+2] for i in range(n_circles)])
                    
                    if validate_packing(centers, radii):
                        best_sum = current_sum
                        best_centers = centers
                        best_radii = radii
                        best_run = i
        except Exception as e:
            print(f"Optimization failed at run {i}: {e}")
            continue

    if best_centers is None:
        # Fallback to simple grid if optimization failed
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        r = 0.05
        idx = 0
        y = 0.15
        counts = [6, 5, 6, 5, 4]
        spacing_x = 0.15
        spacing_y = 0.15
        for r_idx, count in enumerate(counts):
            start_x = (1.0 - (count-1)*spacing_x) / 2
            for c in range(count):
                if idx < 26:
                    x = start_x + c * spacing_x
                    if r_idx % 2 == 1: x += spacing_x/2
                    centers[idx] = [x, y]
                    radii[idx] = r
                    idx += 1
            y += spacing_y
        best_sum = np.sum(radii)

    return best_centers, best_radii, best_sum

# Validation function as provided
import numpy as np

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
