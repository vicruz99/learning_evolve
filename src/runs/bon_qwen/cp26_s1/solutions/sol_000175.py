# sol_000175 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b137705a) state=c1ba8481 sum of radii=0.088546 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # Helper function to calculate penalty
    def calculate_penalty(variables, n):
        # Reshape variables: (n, 3) -> (n, 2) centers, (n,) radii
        # Variables order: x1, y1, r1, x2, y2, r2, ...
        centers = variables.reshape(-1, 3)[:, :2]
        radii = variables.reshape(-1, 3)[:, 2]
        
        penalty = 0.0
        
        # 1. Boundary constraints
        # Circle must be inside [0, 1] x [0, 1]
        # x - r >= 0 => r - x <= 0
        # x + r <= 1 => r + x - 1 <= 0
        # Same for y
        
        # Vectorized boundary check
        # Left wall: r - x
        left_violations = np.maximum(0, radii - centers[:, 0])
        # Right wall: r + x - 1
        right_violations = np.maximum(0, radii + centers[:, 0] - 1)
        # Bottom wall: r - y
        bottom_violations = np.maximum(0, radii - centers[:, 1])
        # Top wall: r + y - 1
        top_violations = np.maximum(0, radii + centers[:, 1] - 1)
        
        boundary_penalty = np.sum(left_violations**2 + right_violations**2 + 
                                  bottom_violations**2 + top_violations**2)
        
        # 2. Overlap constraints
        # dist(c_i, c_j) >= r_i + r_j
        # Violation: r_i + r_j - dist > 0
        
        # Compute all pairwise distances
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Radii sum matrix
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # We only care about upper triangle (i < j)
        # Create a mask for upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Calculate violations
        violations = np.maximum(0, radii_sum - dists)
        overlap_penalty = np.sum(violations[mask]**2)
        
        return boundary_penalty + overlap_penalty

    def objective_function(variables, n):
        # Objective: Minimize -sum(radii) + lambda * penalty
        # We want to maximize sum(radii), so minimize -sum(radii)
        centers = variables.reshape(-1, 3)[:, :2]
        radii = variables.reshape(-1, 3)[:, 2]
        
        obj = -np.sum(radii)
        
        # Adaptive penalty factor? 
        # A high penalty ensures constraints are satisfied.
        # If penalty is too high, gradient might be dominated.
        # If too low, constraints might be violated.
        # Since we validate at the end, we can balance this.
        # But for a valid packing, penalty must be 0.
        # Let's use a large multiplier.
        penalty = calculate_penalty(variables, n)
        
        # If penalty is very small (near 0), we might want to prioritize radius sum?
        # But strictly, we need penalty = 0.
        # Using a large weight ensures we stay in feasible region.
        return obj + 1000.0 * penalty

    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    bounds = [(lb, ub) for lb, ub in bounds]

    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    # Try multiple initial configurations
    # Configuration 1: Grid based
    # Configuration 2: Hexagonal based
    # Configuration 3: Random

    configs = []
    
    # 1. Grid 5x5 + 1 in corner
    centers1 = []
    for i in range(5):
        for j in range(5):
            centers1.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    centers1.append([0.01, 0.01]) # 26th circle
    centers1 = np.array(centers1)
    configs.append(centers1)

    # 2. Hexagonal-like packing
    # 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 6 circles? No, width constraint.
    # Let's try 6, 5, 6, 5, 4 pattern?
    # Or just 5 rows of 5 and 1 extra.
    # Let's make a slightly denser grid.
    centers2 = []
    r_est = 0.09
    # 5 rows
    for row in range(6):
        # Alternate shift
        shift = 0.05 if row % 2 == 1 else 0.0
        y = 0.1 + row * (r_est * 2 * 0.866) # vertical spacing sqrt(3)*r approx 1.732r, but 2r*0.866
        # Number of circles in row
        if row < 5:
            count = 5
        else:
            count = 1 # Just to fill 26? No, 5*5=25.
            # Actually let's just distribute 26 circles in 6 rows: 5, 5, 5, 5, 4, 2?
            # Let's just do 5, 5, 5, 5, 5, 1
            pass 
        
    # Let's construct a specific layout for 26
    # 5 rows of 5 circles is 25.
    # Add 1 circle in a gap or corner.
    # Let's just use the grid 5x5 and add one.
    # But we need to optimize, so initial guess doesn't have to be perfect.
    # Let's generate random centers with some spacing.
    
    # Re-generate centers2 as a perturbed grid
    np.random.seed(42)
    centers2 = np.zeros((26, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers2[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
            idx += 1
    # 26th circle
    centers2[25] = [0.05, 0.05]
    centers2 += np.random.uniform(-0.02, 0.02, centers2.shape)
    # Clip
    centers2 = np.clip(centers2, 0.05, 0.95)
    configs.append(centers2)

    # 3. Random configuration
    centers3 = np.random.uniform(0.1, 0.9, (26, 2))
    configs.append(centers3)

    for init_centers in configs:
        # Initialize radii
        # Start with small radii to ensure no initial penalty
        # Estimate max radius ~ 0.1. Start with 0.02
        init_radii = np.full(n, 0.02)
        
        variables = np.zeros(n * 3)
        for i in range(n):
            variables[3*i] = init_centers[i, 0]
            variables[3*i+1] = init_centers[i, 1]
            variables[3*i+2] = init_radii[i]
        
        # Optimization
        res = minimize(objective_function, variables, args=(n,), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-9})
        
        # Extract result
        res_vars = res.x
        res_centers = res_vars.reshape(-1, 3)[:, :2]
        res_radii = res_vars.reshape(-1, 3)[:, 2]
        
        # Check if valid (low penalty)
        p = calculate_penalty(res_vars, n)
        if p < 1e-6: # Feasible
            current_sum = np.sum(res_radii)
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_centers = res_centers.copy()
                best_radii = res_radii.copy()
        else:
            # Even if penalty > 0, maybe we can clamp radii to satisfy constraints?
            # But we need to return valid packing.
            # If penalty is small, maybe just reduce radii slightly?
            # Let's check overlap and boundary.
            # If invalid, ignore.
            pass

    # Fallback: If no valid packing found (unlikely with small radii start),
    # generate a trivial valid packing (small circles)
    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, (26, 2))
        best_radii = np.full(26, 0.01)
        best_sum_radii = 0.26

    # Final validation check (debug)
    # If somehow invalid, try to shrink radii
    # But with penalty method and low threshold, should be valid.
    # Just to be safe, verify overlaps.
    
    # Re-calculate overlaps manually to be sure
    n = len(best_radii)
    valid = True
    # Boundary check
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                if dist < best_radii[i] + best_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid: break

    if not valid:
        # Try to repair by shrinking radii
        # Simple heuristic: reduce all radii by a factor
        factor = 0.95
        while not valid:
            best_radii *= factor
            # Check again
            valid = True
            for i in range(n):
                x, y = best_centers[i]
                r = best_radii[i]
                if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                        if dist < best_radii[i] + best_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid: break
            factor *= 0.95 # Reduce more
            if factor < 0.01: break # Prevent infinite loop

    return best_centers, best_radii, np.sum(best_radii)
