# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 27de0ea1) state=10d4f69f sum of radii=1.475040 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_packing(n_circles):
    """Generates a hexagonal lattice initialization for circle packing."""
    centers = np.zeros((n_circles, 2))
    r = 0.05  # Start with a small radius to ensure validity
    
    # Estimate grid dimensions
    cols = int(np.ceil(np.sqrt(n_circles * 1.1)))
    rows = int(np.ceil(n_circles / cols))
    
    # Spacing based on hexagonal packing
    spacing_x = 2 * r
    spacing_y = np.sqrt(3) * r
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < n_circles:
                x = spacing_x * col + (row % 2) * r
                y = spacing_y * row
                centers[idx] = [x, y]
                idx += 1
                
    return centers, r

def generate_grid_packing(n_circles):
    """Generates a grid-based initialization, optimized for the square container."""
    centers = np.zeros((n_circles, 2))
    # 5x5 grid is very efficient for n=25. For 26, we perturb.
    side = int(np.ceil(np.sqrt(n_circles)))
    step = 1.0 / (side + 1)
    
    idx = 0
    for r in range(side):
        for c in range(side):
            if idx < n_circles:
                centers[idx] = [(c + 1) * step, (r + 1) * step]
                idx += 1
                
    # Assign a safe initial radius
    return centers, 0.08

def calculate_max_radii(centers):
    """Calculates the maximum possible radius for each circle given fixed centers."""
    n = len(centers)
    radii = np.full(n, 0.5) # Start with max possible
    
    for i in range(n):
        # Distance to boundaries
        dist_boundary = min(centers[i, 0], 1.0 - centers[i, 0], 
                            centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = dist_boundary
        
        # Distance to other centers (must be >= r_i + r_j)
        # We iterate to find the tightest constraint
        # Since r_j is also variable, this is an approximation if we do it sequentially.
        # However, for a "push out" step, we assume neighbors are fixed at their current max?
        # A better way for fixed centers is to solve for r_i = min(dist(j) - r_j) / 2?
        # But let's just use the geometric constraint: dist >= r_i + r_j.
        # If we want to maximize r_i, we are limited by neighbors.
        # If neighbors are fixed, r_i <= dist - r_j.
        # This is tricky if r_j is also changing.
        # Let's assume a static "capacity" check: 
        # r_i <= dist(i,j) / 2 is a safe lower bound on capacity, but we want exact.
        # Let's use a simpler iterative clamp:
        pass

    # Iterative refinement for radii given centers
    for _ in range(5): # A few passes
        for i in range(n):
            # Boundary constraint
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                        centers[i, 1], 1.0 - centers[i, 1])
            
            # Neighbor constraints
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    # dist >= r_i + r_j  =>  r_i <= dist - r_j
                    # But r_j is also unknown. 
                    # In a fixed-center step, we can't simply set r_i based on r_j if r_j is changing.
                    # However, if we assume we just want to check feasibility, 
                    # or we can use a relaxation: r_i + r_j <= dist.
                    # If we fix radii from previous step, we can update.
                    pass
    return radii

def optimize_packing():
    """
    Main optimization function to pack 26 circles in a unit square.
    """
    n = 26
    
    # 1. Initialization
    # Try multiple strategies and pick the best sum
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    strategies = [generate_grid_packing, generate_hexagonal_packing]
    
    for strat in strategies:
        centers, init_r = strat(n)
        # Normalize/Scale to fit in square if necessary
        # The generation functions should output centers in [0,1] roughly.
        # Let's ensure they are valid.
        
        # Initial Radius Assignment (Conservative)
        radii = np.full(n, 0.01) 
        
        # Expand radii as much as possible for these centers
        # Simple iterative expansion
        for _ in range(20):
            for i in range(n):
                max_r = 0.5
                # Boundaries
                max_r = min(max_r, centers[i, 0], 1.0 - centers[i, 0],
                            centers[i, 1], 1.0 - centers[i, 1])
                # Neighbors
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(centers[i] - centers[j])
                        # r_i <= dist - r_j. 
                        # We use current r_j as a lower bound for neighbor size.
                        # Actually, to be safe, we can assume r_j is at least its current value.
                        # But to grow, we should assume r_j might shrink? 
                        # No, we are growing.
                        # A safer bound: r_i <= dist / 2 (if all equal).
                        # Better: r_i <= dist - radii[j]
                        if radii[j] > 0:
                            max_r = min(max_r, dist - radii[j])
                
                if max_r < 0: max_r = 0
                radii[i] = max(0.01, max_r)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    centers = best_centers
    radii = best_radii

    # 2. Sequential Optimization (Hill Climbing / Perturbation)
    # We want to maximize sum(radii). 
    # We can move centers to relieve the tightest constraints.
    
    # Calculate gradients or forces?
    # Let's use a simple force-directed relaxation.
    # Force proportional to overlap or inverse distance to constraint.
    
    # We will run a loop for a fixed number of iterations
    n_iter = 500
    step_size = 0.01
    
    for iteration in range(n_iter):
        # Calculate current "tightness" for each circle
        # Tightness = 1 - (dist / (r_i + r_j)) for neighbors, or similar
        # We want to increase radii, so we identify constraints that limit radii.
        
        # Identify the circle with the smallest "slack"
        # Slack for circle i: min_j (dist(i,j) - r_i - r_j)
        
        slacks = np.full(n, 1e9)
        constraints_j = np.full(n, -1, dtype=int)
        constraints_boundary = np.zeros(n, bool)
        
        for i in range(n):
            # Boundary slack
            slack_b = min(centers[i, 0] - radii[i], 1.0 - centers[i, 0] - radii[i],
                          centers[i, 1] - radii[i], 1.0 - centers[i, 1] - radii[i])
            if slack_b < slacks[i]:
                slacks[i] = slack_b
                constraints_boundary[i] = True
            
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                slack = dist - radii[i] - radii[j]
                if slack < slacks[i]:
                    slacks[i] = slack
                    constraints_j[i] = j
                if slack < slacks[j]:
                    slacks[j] = slack
                    constraints_j[j] = i
        
        # If any slack is very negative (overlap), we must move apart.
        # If slack is positive, we can potentially grow radii.
        
        # Update Radii: Grow if slack > 0
        # We can grow all circles by a factor proportional to their slack?
        # Or just increase the minimum radius?
        # Let's try to equalize radii? No, sum is maximized when equal, but constraints vary.
        
        # Simple growth step:
        # If slack > epsilon, increase radius.
        min_slack = np.min(slacks)
        
        if min_slack > 1e-4:
            # We have room to grow. Let's try to grow the smallest circles?
            # Or all.
            growth = min_slack * 0.5 # Conservative growth
            for i in range(n):
                radii[i] += growth
        else:
            # Tight packing. Move centers to increase slack.
            # Find the most constrained circle (lowest slack)
            idx = np.argmin(slacks)
            
            # Apply force to idx to move away from constraint
            # If constraint is boundary, move towards center?
            if constraints_boundary[idx]:
                # Determine which boundary
                x, y = centers[idx]
                r = radii[idx]
                if x - r == slacks[idx]:
                    centers[idx, 0] += step_size * 0.1 # Move right
                elif 1.0 - x - r == slacks[idx]:
                    centers[idx, 0] -= step_size * 0.1 # Move left
                elif y - r == slacks[idx]:
                    centers[idx, 1] += step_size * 0.1 # Move up
                else:
                    centers[idx, 1] -= step_size * 0.1 # Move down
            else:
                # Constraint with another circle j
                j = constraints_j[idx]
                if j >= 0:
                    # Push idx and j apart
                    diff = centers[idx] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        dir_vec = diff / dist
                        # Move idx away from j
                        centers[idx] += dir_vec * step_size * 0.05
                        # Move j away from idx (optional, but helps)
                        centers[j] -= dir_vec * step_size * 0.05

        # Clamp centers to [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Recalculate valid radii based on new centers to ensure validity
        # This acts as a projection step
        for i in range(n):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                        centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, dist - radii[j]) # This assumes r_j is fixed for this check
            radii[i] = max(1e-5, max_r) # Ensure positive

    # Final refinement using scipy optimization for a few steps
    # Variables: centers (flattened)
    # Objective: -sum(radii)
    # Constraints: Non-overlap, Boundary.
    # This is hard to formulate directly in scipy without complex callbacks.
    # The iterative method above should be sufficient.
    
    # One last check to ensure validity and calculate sum
    # Project radii to be valid
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(centers[i] - centers[j])
                # Valid r_i must satisfy r_i + r_j <= dist
                # If we treat r_j as fixed, r_i <= dist - r_j.
                # But we want to ensure global validity.
                # Let's just enforce r_i <= dist / 2 as a safety if we are unsure?
                # No, let's trust the iterative process but clamp strictly.
                if radii[j] > 0:
                     max_r = min(max_r, dist - radii[j])
        radii[i] = max(0.0, max_r)
        
    # If radii became 0 due to conflicts, we have a bad config.
    # But with 26 circles, it should be fine.
    
    # To improve sum, we can try to "equalize" radii slightly?
    # If one is much smaller, it's a bottleneck.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the packing optimization and returns the result.
    """
    centers, radii, sum_radii = optimize_packing()
    return centers, radii, sum_radii

# Helper to test locally if needed, but run_packing is the entry point.
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation
    valid = True
    n = len(r)
    for i in range(n):
        if r[i] < 0: valid = False
        if c[i, 0] - r[i] < -1e-9 or c[i, 0] + r[i] > 1 + 1e-9: valid = False
        if c[i, 1] - r[i] < -1e-9 or c[i, 1] + r[i] > 1 + 1e-9: valid = False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((c[i]-c[j])**2))
            if dist < r[i] + r[j] - 1e-9:
                valid = False
    print(f"Valid: {valid}")
