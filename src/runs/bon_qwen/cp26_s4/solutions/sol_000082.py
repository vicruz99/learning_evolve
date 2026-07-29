# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e5887d00) state=84514b96 sum of radii=0.993110 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float representing the sum of all radii
    """
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We attempt to fit a hexagonal pattern. 
    # Approximate spacing for 26 circles. 
    # A 5x5 grid has 25 circles. We add one.
    # Let's try a 6x5 staggered grid or similar.
    # Or simply a perturbed grid.
    
    # Initial centers setup
    centers = np.zeros((n, 2))
    
    # Fill a 5x5 grid first (25 circles)
    # Spacing 0.1 gives radius 0.1 exactly at boundaries for 5x5
    # But we want to optimize, so we start slightly looser or just at grid points.
    # Let's use a spacing that fits 5 circles: 1.0 / 4 = 0.25? No.
    # For 5 circles, we need 4 gaps. If gaps are 2r, and 2 radii at ends...
    # Actually, just place them at (0.1, 0.1) to (0.9, 0.9) with step 0.2
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                centers[idx, 0] = 0.1 + i * 0.2
                centers[idx, 1] = 0.1 + j * 0.2
                idx += 1
    
    # Place the 26th circle. Maybe in the center? (0.5, 0.5) is taken.
    # Shift slightly or place in a gap.
    # Let's place it at (0.5, 0.05) or similar if space permits, 
    # but optimization will move it.
    if idx < n:
        centers[idx, 0] = 0.5
        centers[idx, 1] = 0.5
        # Perturb to avoid exact overlap with (0.5, 0.5) if it existed, 
        # but grid was 0.1, 0.3, 0.5, 0.7, 0.9. 
        # (0.5, 0.5) is occupied by center index 12 (3rd row, 3rd col).
        # Let's move it to a gap, e.g., (0.2, 0.2) -> center of 4 circles.
        centers[idx, 0] = 0.2
        centers[idx, 1] = 0.2

    # Initial small radii to ensure validity at start
    radii = np.full(n, 0.01)

    # 2. Iterative Optimization
    # We alternate between optimizing radii (given centers) and centers (given radii).
    
    num_iterations = 1000
    learning_rate = 0.005
    damping = 0.95
    
    for iteration in range(num_iterations):
        
        # --- Step A: Optimize Radii using Linear Programming ---
        # Maximize sum(r_i)
        # Subject to:
        # 1. r_i + r_j <= dist(c_i, c_j) for all i < j
        # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i (Boundary)
        # 3. r_i >= 0
        
        # Distance matrix
        # Compute pairwise distances
        # d[i, j] = distance between i and j
        # We only need lower triangle or upper triangle for constraints
        
        # Constraints matrix A_ub @ x <= b_ub
        # Variables x = [r_0, ..., r_25]
        
        constraints_A = []
        constraints_b = []
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            max_r_x = min(x, 1.0 - x)
            max_r_y = min(y, 1.0 - y)
            limit = max_r_x if max_r_x < max_r_y else max_r_y
            
            # r_i <= limit
            # -r_i >= -limit  =>  r_i <= limit
            # Row: [0, ..., 1, ..., 0] <= limit
            row = np.zeros(n)
            row[i] = 1.0
            constraints_A.append(row)
            constraints_b.append(limit)
            
        # Overlap constraints: r_i + r_j <= dist
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # r_i + r_j <= dist
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                constraints_A.append(row)
                constraints_b.append(dist)
        
        constraints_A = np.array(constraints_A)
        constraints_b = np.array(constraints_b)
        
        # Objective: Maximize sum(r) => Minimize -sum(r)
        c_obj = -np.ones(n)
        
        # Bounds for r_i >= 0
        bounds = [(0, None) for _ in range(n)]
        
        try:
            res = linprog(c_obj, A_ub=constraints_A, b_ub=constraints_b, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
            else:
                # Fallback: keep current radii or solve for max valid radii greedily
                # This shouldn't happen often if centers are valid
                pass
        except Exception:
            pass

        # --- Step B: Optimize Centers using Force-Directed Layout ---
        # Given radii, we want to move centers to increase radii in future steps.
        # Heuristic: Apply repulsive forces between circles that are touching or close.
        # The force magnitude is related to how "tight" the constraint is.
        # Slack = dist - (r_i + r_j). If slack is small/zero, repel strongly.
        
        forces = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                
                if dist < 1e-9:
                    # Avoid division by zero, push randomly
                    diff = np.random.rand(2) * 0.01
                    dist = np.linalg.norm(diff)
                
                required_dist = radii[i] + radii[j]
                overlap = required_dist - dist
                
                if overlap > 0:
                    # Circles are overlapping. Strong repulsion.
                    # Force proportional to overlap
                    repulsion_strength = 10.0
                else:
                    # Not overlapping, but maybe touching.
                    # Apply a soft repulsion to keep them apart, scaling with 
                    # how close they are to touching.
                    # We want to create space.
                    # Force ~ 1/dist^2 or similar?
                    # Let's use a force that is active when dist is small.
                    # But we specifically want to expand the "tight" constraints.
                    # If dist is exactly r_i + r_j, we are constrained.
                    # Let's push apart if dist < r_i + r_j + epsilon.
                    
                    gap = dist - required_dist
                    if gap < 0.05: # If within 0.05 distance of touching
                        # Push apart proportional to 1/gap? No, gap can be negative.
                        # Let's just push if dist is small relative to radii sum
                        # Or just use a standard repulsive force.
                        # Force magnitude = k / dist^2
                        repulsion_strength = 0.5
                    else:
                        repulsion_strength = 0

                # Apply force
                # Vector direction
                direction = diff / dist
                
                # Force magnitude
                if overlap > 0:
                    f_mag = overlap * 5.0 # Strong spring
                else:
                    f_mag = 0 # No force if not overlapping? 
                    # Actually, to maximize sum of radii, we want circles to be 
                    # as far apart as possible, constrained by boundaries.
                    # But simply maximizing distance pushes them to corners.
                    # We only need to resolve overlaps. The LP step handles the size.
                    # If we just resolve overlaps, we might not improve the packing density.
                    # However, the LP step finds max radii for CURRENT centers.
                    # If centers are suboptimal, radii will be small.
                    # Moving centers apart generally allows larger radii.
                    # So applying a generic repulsive force (like charge) helps.
                    f_mag = 0.1 / (dist * dist)

                forces[i] += direction * f_mag
                forces[j] -= direction * f_mag

            # Boundary forces: push back if too close to wall
            x, y = centers[i]
            r = radii[i]
            # Check boundaries
            # We need center to be at least r away.
            # If x < r, force right.
            margin = 0.0 # We can be exactly on boundary
            
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            elif x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 10.0
                
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            elif y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 10.0

        # Update centers
        centers += learning_rate * forces
        
        # Dampen movement to stabilize
        # Actually, just clamping is safer
        # But we must respect that r might change, so we can't strictly clamp to [0,1]
        # without considering r. But centers must be in [0,1].
        # Wait, centers must be in [0,1]. Radii constraints handle the rest.
        # But the LP step uses centers to define bounds.
        # If center goes outside [0,1], LP might behave weirdly (negative max_r).
        # So keep centers strictly inside [0,1].
        
        centers = np.clip(centers, 1e-6, 1 - 1e-6)

    # Final LP solve to get exact maximal radii for final centers
    constraints_A = []
    constraints_b = []
    
    for i in range(n):
        x, y = centers[i]
        limit = min(x, 1.0 - x, y, 1.0 - y)
        row = np.zeros(n)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(limit)
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints_A.append(row)
            constraints_b.append(dist)
            
    constraints_A = np.array(constraints_A)
    constraints_b = np.array(constraints_b)
    c_obj = -np.ones(n)
    bounds = [(0, None) for _ in range(n)]
    
    res = linprog(c_obj, A_ub=constraints_A, b_ub=constraints_b, bounds=bounds, method='highs')
    if res.success:
        radii = res.x
    else:
        # Fallback to small radii if LP fails (unlikely)
        radii = np.full(n, 0.01)
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
