# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3353d097) state=042f03a8 sum of radii=2.592939 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses scipy.optimize to find the optimal configuration.
    """
    n = 26
    
    # Initialize variables
    # We will optimize for centers (x, y) and radii r
    # Variables vector: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Shape: 78
    
    # Initial guess: A 5x5 grid with one extra circle
    # Grid positions for 25 circles
    grid_coords = []
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            grid_coords.append((x, y))
    
    # Add 26th circle in a gap, e.g., center of first hole
    # Hole at (0.2, 0.2) relative to grid origin 0.1? 
    # Centers are 0.1, 0.3... Hole center between (0.1,0.1) and (0.3,0.3) is (0.2, 0.2)
    grid_coords.append((0.2, 0.2))
    
    # Initial radii: small value to ensure valid start
    # Actually, if we start with r=0.05, circles at 0.1, 0.3 etc might overlap if spacing is 0.2 (diameter 0.1). 
    # 0.05 is safe.
    init_radii = np.full(n, 0.05)
    
    # Construct initial vector
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = grid_coords[i][0]
        x0[3*i+1] = grid_coords[i][1]
        x0[3*i+2] = init_radii[i]
        
    # Bounds: 
    # x in [0, 1], y in [0, 1], r in [0, 0.5] (max possible radius)
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Constraints
    constraints = []
    
    # 1. Boundary constraints: r <= x <= 1-r  =>  x - r >= 0,  x + r <= 1
    # 2. Non-overlap: dist >= r_i + r_j  =>  dist^2 - (r_i + r_j)^2 >= 0
    
    def constraint_boundary_center(v, i):
        # For circle i
        xi = v[3*i]
        yi = v[3*i+1]
        ri = v[3*i+2]
        # x - r >= 0
        # y - r >= 0
        # x + r <= 1  =>  1 - x - r >= 0
        # y + r <= 1  =>  1 - y - r >= 0
        return np.array([xi - ri, yi - ri, 1.0 - xi - ri, 1.0 - yi - ri])

    # Add boundary constraints for each circle
    # Note: minimize expects constraints to be functions returning values >= 0
    # We can add them as separate constraints or vectorized?
    # SLSQP supports multiple constraints.
    # Let's add them individually or in groups.
    # For efficiency, let's just add a callback or define them in the objective with penalty?
    # No, constraints are better.
    
    # However, adding 26 * 4 = 104 boundary constraints plus 26*25/2 = 325 overlap constraints
    # might be slow.
    # Alternative: Penalty method in objective.
    
    # Let's try a penalty method first for simplicity and speed, 
    # but with a hard constraint check at the end.
    # Or use 'trust-constr' which handles many constraints.
    
    # Let's stick to SLSQP with constraints defined via a function that returns a vector.
    
    def get_constraints(v):
        con_vals = []
        # Boundary constraints
        for i in range(n):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            con_vals.append(xi - ri)
            con_vals.append(yi - ri)
            con_vals.append(1.0 - xi - ri)
            con_vals.append(1.0 - yi - ri)
        
        # Overlap constraints
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # dist_sq - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist_sq = (ri + rj)**2
                con_vals.append(dist_sq - min_dist_sq)
        
        return np.array(con_vals)

    # Define constraints for SLSQP
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    # Objective: Maximize sum of radii -> Minimize -sum(r_i)
    def objective(v):
        r_sum = 0.0
        for i in range(n):
            r_sum += v[3*i + 2]
        return -r_sum

    # Run optimization
    # We might need multiple restarts to escape local optima.
    # Let's try one run with a good initial guess.
    # The initial guess is a grid, which is feasible (with r=0.05).
    # Wait, with r=0.05, boundaries: 0.1-0.05 = 0.05 >= 0. OK.
    # Overlap: dist 0.2, sum radii 0.1. 0.04 - 0.01 > 0. OK.
    
    # Try to optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Check if successful
    if not result.success:
        # If failed, try a randomized restart or just return best found
        pass
        
    # Extract results
    best_v = result.x
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = best_v[3*i]
        centers[i, 1] = best_v[3*i+1]
        radii[i] = best_v[3*i+2]
        
    sum_radii = np.sum(radii)
    
    # Validate
    # We can run the provided validation logic locally to ensure correctness
    # But the problem statement says we must return valid results.
    # The optimizer with constraints should guarantee validity.
    # However, numerical precision might be an issue.
    
    # Let's enforce constraints strictly if close
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # Clamp radius to fit in square
        max_r_x = min(x, 1-x)
        max_r_y = min(y, 1-y)
        r = min(r, max_r_x, max_r_y)
        # Ensure non-negative
        r = max(0.0, r)
        centers[i, 0] = x
        centers[i, 1] = y
        radii[i] = r
        
    # Re-check overlaps and shrink if necessary (simple fix)
    # If circles overlap, reduce radius of the larger one or both.
    # This is a post-processing step.
    for _ in range(10): # Iterate a few times
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                required_dist = radii[i] + radii[j]
                if dist < required_dist - 1e-9:
                    # Overlap detected, reduce radii
                    # Reduce proportionally or just cut the overlap
                    overlap = required_dist - dist
                    # Reduce both by half overlap
                    radii[i] -= overlap / 2
                    radii[j] -= overlap / 2
                    # Ensure non-negative
                    radii[i] = max(0, radii[i])
                    radii[j] = max(0, radii[j])

    # Recalculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper to validate locally (optional, but good for debugging if I could run it)
# But I must output the function.

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    # Validate
    import numpy as np # re-import inside if needed, but already imported
    
    # Check constraints manually
    valid = True
    n = 26
    if np.isnan(centers).any() or np.isnan(radii).any():
        valid = False
    for i in range(n):
        if radii[i] < 0: valid = False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            valid = False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-12:
                valid = False
    
    print(f"Valid: {valid}")
