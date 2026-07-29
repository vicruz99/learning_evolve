# sol_000316 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5cd869be) state=d74ece19 sum of radii=2.515680 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float, the sum of radii
    """
    n = 26
    
    # 1. Initialization
    # Create a grid of points to start with. 
    # A 5x5 grid gives 25 points. We add one more in a gap.
    # Using spacing 0.2 centered at 0.1, 0.3, 0.5, 0.7, 0.9
    coords = np.linspace(0.1, 0.9, 5)
    grid_points = []
    for x in coords:
        for y in coords:
            grid_points.append([x, y])
    
    # We have 25 points. Add one more. 
    # The center (0.5, 0.5) is occupied. 
    # Let's try a point in a gap, e.g., (0.2, 0.2) which is equidistant to (0.1,0.1), (0.3,0.1), etc.
    # Distance from (0.2, 0.2) to (0.1, 0.1) is sqrt(0.02) approx 0.141.
    # If radius is small, this is valid.
    # Let's just add a point at (0.2, 0.2).
    # However, to be safe and symmetric, maybe we can use a hexagonal packing logic 
    # or just shuffle and pick. 
    # Let's stick to the grid + one point.
    # To make it more robust, let's generate a slightly randomized grid or use a specific pattern.
    # A 5x5 grid is quite rigid. Let's try to place 26 points more evenly.
    # Maybe 6 points in some rows?
    # Row y=0.1: x in 0.1..0.9 (5 pts)
    # Row y=0.3: x in 0.2..0.8 (5 pts) -- shifted
    # This creates hexagonal packing.
    # Let's generate hex grid points.
    
    centers = []
    spacing = 0.2 # Initial spacing
    # Hexagonal packing rows
    # y spacing is spacing * sqrt(3)/2
    dy = spacing * np.sqrt(3) / 2
    
    # We want to fit in [0, 1]. 
    # Let's just generate points and filter/adjust.
    # Or simpler: Use the grid method but ensure validity.
    
    # Let's go back to the 5x5 grid + 1 strategy, it's easier to ensure initial validity.
    # Grid points
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    initial_centers = []
    for x in xs:
        for y in ys:
            initial_centers.append([x, y])
    
    # Add 26th point. (0.2, 0.2) is a good candidate.
    # Check distance to nearest grid points (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3).
    # Dist is sqrt(0.1^2 + 0.1^2) = 0.1414.
    # If we set initial radius very small, say 0.01, it's valid.
    initial_centers.append([0.2, 0.2])
    
    centers = np.array(initial_centers)
    
    # 2. Iterative Optimization
    # We will iterate: 
    # a) Solve LP for radii given centers
    # b) Move centers apart based on active constraints
    
    max_iter = 100
    best_sum_r = 0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    # Tolerance for convergence
    tol = 1e-5
    
    # Precompute index pairs for distance constraints
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    # Helper to solve LP for radii
    def solve_radii(centers):
        # Variables: r_0, ..., r_25
        # Maximize sum(r) => Minimize -sum(r)
        c_obj = -np.ones(n)
        
        # Constraints: A_ub @ r <= b_ub
        # 1. Distance constraints: r_i + r_j <= dist(i, j)
        # 2. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        
        n_pairs = len(pairs)
        n_bounds = n * 4
        A_ub = np.zeros((n_pairs + n_bounds, n))
        b_ub = np.zeros(n_pairs + n_bounds)
        
        idx = 0
        for i, j in pairs:
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
        for i in range(n):
            # r_i <= x_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = centers[i, 0]
            idx += 1
            # r_i <= 1 - x_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = 1.0 - centers[i, 0]
            idx += 1
            # r_i <= y_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = centers[i, 1]
            idx += 1
            # r_i <= 1 - y_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = 1.0 - centers[i, 1]
            idx += 1
            
        bounds = [(0, None)] * n
        
        # Use highs method for speed
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            # Fallback to small radii if LP fails (should not happen with valid centers)
            return np.full(n, 0.01)

    # Initial radii
    radii = solve_radii(centers)
    current_sum = np.sum(radii)
    
    # Optimization loop
    for iteration in range(max_iter):
        # 1. Solve for optimal radii
        radii = solve_radii(centers)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum_r:
            best_sum_r = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
        
        # Check convergence
        if current_sum > 2.635: # Target reached
             break
            
        # 2. Compute forces to move centers
        # We want to increase distances between circles that are touching.
        # Touching condition: r_i + r_j approx dist(i, j)
        
        forces = np.zeros_like(centers)
        
        # Threshold for "touching"
        touch_thresh = 1e-4
        
        for i, j in pairs:
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            r_sum = radii[i] + radii[j]
            
            # If they are very close to touching, repel
            if d < r_sum + touch_thresh:
                # Vector from j to i
                diff = centers[i] - centers[j]
                # Normalize
                if d > 1e-9:
                    dir_vec = diff / d
                    # Force magnitude: proportional to how much they overlap or just constant?
                    # Since we want to increase d to allow r to grow, 
                    # we apply a repulsive force.
                    # A simple heuristic: force = 1.0
                    # Or force = (r_sum - d) if overlap, but here d >= r_sum (usually)
                    # Actually, d >= r_sum is guaranteed by LP.
                    # But if d is close to r_sum, constraint is tight.
                    # We push apart.
                    # Force magnitude could be proportional to r_sum (larger circles need more space?)
                    force_mag = 0.5 # constant repulsion
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag
        
        # Boundary forces
        # If r_i is limited by boundary, push center away from boundary
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left boundary x >= r
            if x - r < touch_thresh:
                forces[i, 0] += 0.5
            # Right boundary x <= 1-r
            if 1 - x - r < touch_thresh:
                forces[i, 0] -= 0.5
            # Bottom boundary y >= r
            if y - r < touch_thresh:
                forces[i, 1] += 0.5
            # Top boundary y <= 1-r
            if 1 - y - r < touch_thresh:
                forces[i, 1] -= 0.5
                
        # Apply movement
        # Step size needs to be small enough to not jump over optima
        # But large enough to converge.
        # As radii grow, step size might need to decrease?
        step_size = 0.01 
        
        centers = centers + step_size * forces
        
        # Clip centers to stay within reasonable bounds (0, 1)
        # Actually, centers must be within [0, 1], but radii constraint enforces [r, 1-r].
        # Just clip to [0, 1] for now.
        centers = np.clip(centers, 0.0, 1.0)

    # Final cleanup and validation
    # The iterative process might leave tiny overlaps due to floating point or step size.
    # We should project to a valid state.
    # Re-solve LP one last time to ensure consistency.
    final_radii = solve_radii(centers)
    
    # Check for validity with a small slack
    # If validation fails, we might need to adjust.
    # But linprog guarantees r_i + r_j <= dist.
    # The only issue is if centers moved such that distances decreased?
    # No, we re-solve LP at end.
    # So final_radii is valid for final centers.
    
    # However, the validation function checks strict inequality with tolerance.
    # linprog uses <=.
    # So it should be fine.
    
    # Just to be safe, verify and fix any tiny boundary violations
    for i in range(n):
        x, y = centers[i]
        r = final_radii[i]
        # Enforce bounds strictly
        if x < r: x = r
        if x > 1 - r: x = 1 - r
        if y < r: y = r
        if y > 1 - r: y = 1 - r
        centers[i] = [x, y]
        
    # After moving centers to fix boundaries, radii might need adjustment?
    # Actually, if we clamp center to r, we are at boundary, which is valid.
    # But moving center closer to boundary might increase distance to other circles?
    # Or decrease?
    # If we clamp x to r, we moved center right. Distance to left circles increases, right decreases.
    # This might violate constraints.
    # Safer: Just rely on LP solution. The LP solution satisfies r_i <= x_i etc.
    # So x_i >= r_i is guaranteed.
    # x_i <= 1 - r_i is guaranteed.
    # So centers are valid.
    # The only risk is if we clipped centers in the loop, we might have moved them into invalid spots relative to each other?
    # But we re-solved LP at the end with the final centers.
    # So final_radii is consistent with final centers.
    
    # One edge case: if linprog failed? Unlikely.
    
    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii
