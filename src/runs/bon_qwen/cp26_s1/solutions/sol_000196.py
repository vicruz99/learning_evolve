# sol_000196 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1a220354) state=ed2eb911 sum of radii=2.452607 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
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

def compute_max_radius_for_circle(i, centers, radii):
    """
    Calculates the maximum radius circle i can have without overlapping others
    or crossing boundaries, given fixed positions of all circles.
    """
    x, y = centers[i]
    r_current = radii[i]
    
    # Constraint from boundaries
    max_r_boundary = min(x, 1 - x, y, 1 - y)
    
    # Constraint from other circles
    max_r_neighbors = float('inf')
    for j in range(len(radii)):
        if i == j:
            continue
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        dist = math.sqrt(dx*dx + dy*dy)
        # We need dist >= r_i + r_j  =>  r_i <= dist - r_j
        r_limit = dist - radii[j]
        if r_limit < max_r_neighbors:
            max_r_neighbors = r_limit
            
    return min(max_r_boundary, max_r_neighbors)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- 1. Initialization ---
    # Start with a 5x5 grid pattern but perturb to fit 26 circles.
    # We can place 25 in a grid and 1 in a gap, or use a denser hexagonal-ish layout.
    # Let's try a dense random start first, then refine. 
    # Or better, a structured start: 5 rows of 5, but compressed, plus 1 extra.
    
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.0)
    
    # Initialize with a slightly compressed 5x5 grid + 1 center
    # Grid spacing 0.18 roughly?
    # Let's place them in a hexagonal lattice pattern to start dense
    # Rows: 5, 4, 5, 4, 5, 3? Sum = 26.
    # Let's try: 6 rows.
    # Row 1: 5 circles
    # Row 2: 4 circles (shifted)
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Row 5: 5 circles
    # Row 6: 3 circles
    # Total 26.
    
    row_counts = [5, 4, 5, 4, 5, 3]
    # Approximate radius for this config in unit square?
    # Width for 5 circles: 10r <= 1 -> r <= 0.1
    # Height for 6 rows: (2 + 5*sqrt(3))r approx 10.6r <= 1 -> r <= 0.094
    # Let's initialize with r=0.09
    
    r_init = 0.08
    radii[:] = r_init
    
    idx = 0
    # Vertical spacing for hex packing
    dy = math.sqrt(3) * r_init * 1.01 # slight overlap allowed initially
    # Horizontal spacing
    dx = 2 * r_init * 1.01
    
    for row_idx, count in enumerate(row_counts):
        y = r_init + row_idx * dy
        # Center the row horizontally
        # Total width occupied by row = (count - 1) * 2r + 2r = 2*count*r
        # But we want them centered in [0, 1]
        # Actually, let's just space them evenly in [r_init, 1-r_init]
        
        if count > 0:
            x_start = r_init
            x_end = 1 - r_init
            if count == 1:
                x_positions = [0.5]
            else:
                spacing = (x_end - x_start) / (count - 1)
                x_positions = [x_start + i * spacing for i in range(count)]
            
            # If shifted row (even index in hex packing usually shifted by r), 
            # but here we just center them to maximize space usage roughly.
            # For hex, even rows are shifted. Let's apply a shift for even rows.
            if row_idx % 2 == 1:
                shift = r_init * 0.5 # partial shift
                x_positions = [x + shift for x in x_positions]
                # Keep within bounds
                x_positions = [max(r_init, min(1-r_init, x)) for x in x_positions]

            for x in x_positions:
                if idx < n:
                    centers[idx] = [x, y]
                    idx += 1
    
    # --- 2. Optimization ---
    # We will use a simple iterative relaxation.
    # 1. Maximize radii given centers.
    # 2. Repel circles if they overlap (treat as hard constraints or high penalty).
    # 3. Repeat.
    
    # However, to maximize SUM of radii, we should try to keep radii equal or 
    # slightly adjusted.
    
    # Let's use a gradient-free local search.
    # State: centers, radii.
    # Objective: Sum(radii).
    # Constraints: Validity.
    
    # We can use a "simulated annealing" style approach but with deterministic cooling
    # or just random local moves.
    
    # Better approach:
    # Treat radii as variables. Fix sum of radii = S.
    # Check if feasible. Binary search S?
    # Feasibility check is hard (non-convex).
    
    # Alternative: 
    # Randomly perturb centers and radii. If valid and sum increased, accept.
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(best_radii)
    
    # Initial feasibility fix: shrink radii until valid
    # While invalid, reduce radii slightly
    while not validate_packing(best_centers, best_radii):
        best_radii *= 0.99
        # Also clamp to boundaries
        for i in range(n):
            x, y = best_centers[i]
            best_radii[i] = min(best_radii[i], x, 1-x, y, 1-y)
    
    # Now try to improve
    # Temperature for random search
    temp = 0.05
    step_size = 0.02
    
    # Run optimization loop
    num_iterations = 5000 # Tunable
    
    current_centers = best_centers.copy()
    current_radii = best_radii.copy()
    current_sum = best_sum
    
    # To encourage exploration, we can occasionally reset or use large steps
    # But let's stick to local refinement first.
    
    # We can try to increase radii directly.
    # Strategy: 
    # 1. Pick a circle, try to increase its radius.
    # 2. If it overlaps, push it or neighbors.
    
    # Let's implement a simple "expand and repel" loop.
    
    for iteration in range(num_iterations):
        # Try to increase radii
        # We can try to scale all radii up by a small factor
        factor = 1.0 + 0.001 * (1 - iteration/num_iterations) # annealing expansion
        trial_radii = current_radii * factor
        
        # Check if valid (ignoring overlaps for a moment, just bounds)
        # Actually, overlaps will happen. We need to resolve them.
        
        # Instead of scaling, let's do random moves.
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Move a circle center
            i = random.randint(0, n-1)
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            new_centers = current_centers.copy()
            new_centers[i, 0] += dx
            new_centers[i, 1] += dy
            
            # Clamp to bounds (loose)
            new_centers[i, 0] = max(0, min(1, new_centers[i, 0]))
            new_centers[i, 1] = max(0, min(1, new_centers[i, 1]))
            
            # Check validity with current radii
            if validate_packing(new_centers, current_radii):
                # Check if we can increase radius of this circle or others
                # Accept if valid. We don't gain sum immediately, but might enable future gain.
                # To make progress, we prefer moves that allow radius increase.
                # But simply moving to free space is good.
                current_centers = new_centers
            else:
                # If invalid, maybe we pushed into a neighbor.
                # Revert.
                pass
                
        elif move_type < 0.8:
            # Try to increase radius of a random circle
            i = random.randint(0, n-1)
            # Calculate max possible radius for this circle at current position
            r_max = compute_max_radius_for_circle(i, current_centers, current_radii)
            
            # Try to set radius to something between current and max
            # Or just increase slightly
            delta = random.uniform(0, (r_max - current_radii[i]) * 0.5)
            if delta > 1e-6:
                new_radii = current_radii.copy()
                new_radii[i] += delta
                
                if validate_packing(current_centers, new_radii):
                    current_radii = new_radii
                    current_sum = np.sum(current_radii)
                    if current_sum > best_sum:
                        best_centers = current_centers.copy()
                        best_radii = current_radii.copy()
                        best_sum = current_sum
                        
        else:
            # Global scaling attempt
            # Try to increase all radii by small epsilon
            epsilon = 0.0005
            trial_radii = current_radii + epsilon
            
            # If valid, accept
            if validate_packing(current_centers, trial_radii):
                current_radii = trial_radii
                current_sum = np.sum(current_radii)
                if current_sum > best_sum:
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
                    best_sum = current_sum
            else:
                # If invalid, maybe we can resolve by moving?
                # Complex. Skip for now.
                pass

    # --- 3. Final Refinement (Radius Maximization) ---
    # Given the best centers found, maximize radii.
    # This is an iterative process: fix centers, solve for max radii (LP-like), 
    # then adjust centers to free up space.
    
    # Since N=26 is small, we can do a few passes of "max radius at fixed center"
    # and "perturb center to increase radius".
    
    current_centers = best_centers.copy()
    current_radii = best_radii.copy()
    
    for _ in range(200):
        changed = False
        for i in range(n):
            r_max = compute_max_radius_for_circle(i, current_centers, current_radii)
            if r_max > current_radii[i] + 1e-7:
                # We can increase radius
                # But increasing radius might invalidate others?
                # compute_max_radius_for_circle assumes OTHER radii are fixed.
                # So if we increase r_i, we must ensure we don't overlap.
                # The function calculates limit based on fixed neighbors.
                # So setting r_i = r_max is safe wrt neighbors.
                current_radii[i] = r_max
                changed = True
        
        if not changed:
            break
            
    # After maximizing radii at fixed centers, the packing is "tight" (jammed).
    # We might be able to wiggle centers to increase sum further.
    # Let's run a few more random perturbations accepting only if sum increases.
    
    for _ in range(1000):
        i = random.randint(0, n-1)
        dx = random.gauss(0, 0.01)
        dy = random.gauss(0, 0.01)
        
        trial_centers = current_centers.copy()
        trial_centers[i, 0] += dx
        trial_centers[i, 1] += dy
        trial_centers[i, 0] = max(0, min(1, trial_centers[i, 0]))
        trial_centers[i, 1] = max(0, min(1, trial_centers[i, 1]))
        
        # Calculate max radii for all circles at these new centers
        # Note: This is greedy, might decrease others, but let's see.
        # Actually, better to keep radii fixed and check validity, 
        # then expand.
        
        # Check validity with CURRENT radii
        if validate_packing(trial_centers, current_radii):
            # If valid, try to expand radii
            # Greedy expansion
            new_radii = current_radii.copy()
            for k in range(n):
                r_lim = compute_max_radius_for_circle(k, trial_centers, new_radii)
                new_radii[k] = max(new_radii[k], r_lim) # Keep current or increase
            
            new_sum = np.sum(new_radii)
            if new_sum > np.sum(current_radii) + 1e-6:
                current_centers = trial_centers
                current_radii = new_radii
                
    # Final validation and cleanup
    # Ensure no numerical issues
    current_radii = np.maximum(current_radii, 0)
    for i in range(n):
        current_centers[i, 0] = max(current_radii[i], min(1-current_radii[i], current_centers[i, 0]))
        current_centers[i, 1] = max(current_radii[i], min(1-current_radii[i], current_centers[i, 1]))
        
    # One last pass to clamp radii to fit boundaries exactly
    for i in range(n):
        r_lim = min(current_centers[i,0], 1-current_centers[i,0], 
                    current_centers[i,1], 1-current_centers[i,1])
        current_radii[i] = min(current_radii[i], r_lim)
        
    # Check for overlaps and shrink if necessary
    for _ in range(10):
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
                req_dist = current_radii[i] + current_radii[j]
                if dist < req_dist - 1e-9:
                    overlap = req_dist - dist
                    # Reduce radii to fix
                    reduction = overlap / 2 + 1e-7
                    current_radii[i] = max(0, current_radii[i] - reduction)
                    current_radii[j] = max(0, current_radii[j] - reduction)
                    valid = False
        if valid: break

    final_sum = np.sum(current_radii)
    
    return current_centers, current_radii, float(final_sum)
