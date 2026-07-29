# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b5cb09ab) state=171265bc sum of radii=2.502786 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # Helper function to calculate objective (negative sum of radii + penalties)
    def objective(params):
        # params shape: (3*n,) -> [x1, y1, r1, x2, y2, r2, ...]
        # Reshape to (n, 3)
        pts = params.reshape((n, 3))
        centers = pts[:, :2]
        radii = pts[:, 2]
        
        # Objective: Maximize sum of radii => Minimize -sum(radii)
        obj = -np.sum(radii)
        
        # Penalty coefficient. Needs to be large enough to enforce constraints
        # but not so large that it dominates gradients for valid regions.
        # We can use a dynamic penalty or a fixed large one.
        penalty_weight = 1000.0
        
        # Boundary penalties
        # x - r >= 0  => max(0, r - x)^2
        # x + r <= 1  => max(0, x + r - 1)^2
        # same for y
        
        pen_boundary = 0.0
        for i in range(n):
            x, y, r = pts[i]
            # Left
            if x - r < 0:
                pen_boundary += (x - r)**2
            # Right
            if x + r > 1:
                pen_boundary += (x + r - 1)**2
            # Bottom
            if y - r < 0:
                pen_boundary += (y - r)**2
            # Top
            if y + r > 1:
                pen_boundary += (y + r - 1)**2
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = pts[i, :2]
                c2 = pts[j, :2]
                r1 = pts[i, 2]
                r2 = pts[j, 2]
                
                dist = np.sqrt(np.sum((c1 - c2)**2))
                min_dist = r1 + r2
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        
        return obj + penalty_weight * (pen_boundary + pen_overlap)

    # Helper to generate initial hexagonal-like configuration
    def get_initial_config():
        # We want to place 26 circles.
        # A hexagonal packing with rows of sizes 6, 5, 6, 5, 4 fits well.
        # Let's place them roughly in the square.
        
        centers = []
        # Approximate radius to start with, small enough to fit easily
        r_start = 0.05
        
        # Rows configuration
        # Row 1: 6 circles
        # Row 2: 5 circles (shifted)
        # Row 3: 6 circles
        # Row 4: 5 circles (shifted)
        # Row 5: 4 circles
        
        # We need to fit this in [0,1]x[0,1].
        # Let's compute positions based on a grid and then scale/center.
        
        # Ideal hexagonal coordinates for rows
        # y spacing: sqrt(3)/2 * diameter? No, row spacing is sqrt(3)*r.
        # Let's just place them in a grid and let optimizer fix it.
        # But a valid start is better.
        
        # Let's create a 5x6 grid (30 points) and pick 26?
        # Or just place them manually.
        
        # Pattern:
        # Row 0 (y=0.1): 6 circles
        # Row 1 (y=0.3): 5 circles
        # Row 2 (y=0.5): 6 circles
        # Row 3 (y=0.7): 5 circles
        # Row 4 (y=0.9): 4 circles
        # This is just a guess. Let's make it denser.
        
        # Let's use a standard hexagonal lattice generator logic.
        # But since we don't know optimal r, let's place centers uniformly
        # and set radii small.
        
        # Actually, let's try to fit a hexagonal packing of 26 circles
        # with a specific radius r, calculate bounds, and scale.
        # But simpler: random valid placement or grid.
        
        # Let's try a 5-row structure with specific counts.
        counts = [6, 5, 6, 5, 4]
        y_coords = [0.1, 0.3, 0.5, 0.7, 0.9] # Initial y
        
        for row_idx, count in enumerate(counts):
            y = y_coords[row_idx]
            # Determine x spacing
            # We want to fit 'count' circles in width 1.
            # If they were size 0, we could put them anywhere.
            # Let's space them evenly.
            # If count=6, spacing ~ 1/7? Centers at 1/14, 3/14...
            # Let's just use linspace.
            if row_idx % 2 == 1:
                # Shifted row (odd index 1, 3)
                # To simulate hexagonal, shift by half spacing
                # But let's just center them.
                xs = np.linspace(0.1 + 0.05, 0.9 - 0.05, count) 
            else:
                xs = np.linspace(0.1, 0.9, count)
            
            for x in xs:
                centers.append([x, y])
        
        centers = np.array(centers)
        radii = np.full(n, 0.04) # Start with small radii
        
        # Construct params
        params = np.empty((n, 3))
        params[:, 0] = centers[:, 0]
        params[:, 1] = centers[:, 1]
        params[:, 2] = radii
        
        return params.flatten()

    # Run optimization
    # We might need multiple restarts or a careful initial guess.
    # Let's try one run with a good guess.
    
    # Better initial guess: Hexagonal lattice scaling
    # Let's construct a lattice that fits 26 circles with radius ~0.1
    # and then optimize.
    
    # Let's build a specific pattern that is known to be good for n=26
    # Rows: 5, 6, 5, 6, 4 is not possible (26).
    # 6, 5, 6, 5, 4 = 26.
    
    # Let's construct this pattern geometrically.
    # We want circles of radius r.
    # Width for 6 circles: 12r.
    # Height for 5 rows: 2r + 4*sqrt(3)r.
    # We need 12r <= 1 => r <= 0.0833.
    # We need r(2 + 4*1.732) <= 1 => 8.928r <= 1 => r <= 0.112.
    # So r=0.0833 is limit.
    # But we can deform.
    
    # Let's initialize with r=0.05 and centers in this pattern.
    r_init = 0.05
    centers_init = []
    
    # Row 1: 6 circles
    # x range: [r, 1-r]. Length 1-2r = 0.9.
    # 6 circles diameter 0.1. Total width 0.6. Fits easily.
    # Space them evenly?
    # For hexagonal, row 1 centers at x = r + i*2r? No, that's touching.
    # Let's just place them evenly distributed.
    
    # Pattern:
    # Row 0 (6 circles)
    # Row 1 (5 circles, shifted)
    # Row 2 (6 circles)
    # Row 3 (5 circles, shifted)
    # Row 4 (4 circles)
    
    # Y positions
    # Let's spread them vertically
    y_pos = [0.15, 0.35, 0.55, 0.75, 0.9] # Approx
    # Adjust to be more regular?
    # Let's use a parameter to define y spacing.
    
    # Actually, let's use the optimizer to find the best positions.
    # Just give a reasonable start.
    
    current_params = get_initial_config()
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 1] (actually max r is 0.5)
    bounds = []
    for i in range(n):
        bounds.extend([
            (0.0, 1.0), # x
            (0.0, 1.0), # y
            (0.0, 0.5)  # r
        ])
    
    # Optimization
    # L-BFGS-B is good for bounds.
    # We use the penalty method.
    
    # To improve convergence, we can increase penalty weight gradually?
    # Or just use a fixed large weight.
    
    res = scipy.optimize.minimize(
        objective,
        current_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
    )
    
    # Extract solution
    sol_params = res.x.reshape((n, 3))
    centers_opt = sol_params[:, :2]
    radii_opt = sol_params[:, 2]
    
    # Check if valid and maybe refine
    # The penalty method might leave small violations if penalty not high enough
    # or if stuck.
    # Let's verify and if invalid, we might need to scale down radii slightly.
    
    # However, for the purpose of the function, we return the result.
    # We should ensure the returned solution is valid according to the validation function.
    # The validation function has a tolerance of 1e-12.
    # Our penalty uses squared error, so it might not be strict.
    
    # Let's run a local refinement to strictly satisfy constraints if needed.
    # Or just scale radii down if overlaps exist.
    
    # Check overlaps
    is_valid = True
    # Simple check
    for i in range(n):
        if radii_opt[i] < 0:
            radii_opt[i] = 0
            is_valid = False
        # Boundary check
        if centers_opt[i, 0] < radii_opt[i] or centers_opt[i, 0] > 1 - radii_opt[i]:
            # Clip
            r = radii_opt[i]
            x = centers_opt[i, 0]
            # Project to valid interval [r, 1-r]
            # But this changes center.
            # Better to just reduce radius.
            max_r = min(x, 1-x)
            if radii_opt[i] > max_r:
                radii_opt[i] = max_r
                is_valid = False
        
        if centers_opt[i, 1] < radii_opt[i] or centers_opt[i, 1] > 1 - radii_opt[i]:
            r = radii_opt[i]
            y = centers_opt[i, 1]
            max_r = min(y, 1-y)
            if radii_opt[i] > max_r:
                radii_opt[i] = max_r
                is_valid = False

    # Fix overlaps by reducing radii
    # This is a greedy fix.
    # If circles overlap, reduce the larger one or both.
    # A simple way: calculate max feasible radius for each circle given others fixed?
    # Too complex.
    # Just scaling down all radii slightly if any overlap is detected.
    
    overlap_found = True
    scale_factor = 1.0
    while overlap_found:
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                req_dist = radii_opt[i] + radii_opt[j]
                if dist < req_dist - 1e-12:
                    # Overlap detected
                    # Reduce radii proportionally to fit
                    # New sum of radii <= dist
                    # r_i + r_j = dist
                    # We need to reduce them.
                    # Simple strategy: reduce both by half the overlap?
                    # Or scale both down.
                    excess = req_dist - dist
                    # Distribute excess reduction
                    # If r_i > r_j, reduce r_i more?
                    # Just reduce both by excess/2
                    red = excess / 2 + 1e-6
                    radii_opt[i] = max(0, radii_opt[i] - red)
                    radii_opt[j] = max(0, radii_opt[j] - red)
                    overlap_found = True # Re-check?
                    # Breaking might be needed to avoid infinite loop if logic is flawed
                    # But reducing radii strictly reduces req_dist, so it should converge.
        
        # Re-check boundary after radius reduction?
        # Radii reduction helps boundary constraints too.
        
        # Re-check all pairs? The loop above does it.
        # But changing radii might fix some and not others.
        # The while loop handles it.
        # However, modifying radii inside the loop might be inefficient.
        # Let's just run it a fixed number of times or until stable.
        pass # Logic inside loop handles it? No, loop structure is tricky.
    
    # Better fix:
    # Just scale all radii down by a small factor if invalid.
    # Check validity
    def check_valid(c, r):
        for i in range(len(c)):
            if c[i,0] < r[i] or c[i,0] > 1-r[i] or c[i,1] < r[i] or c[i,1] > 1-r[i]:
                return False
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                dist = np.sqrt(np.sum((c[i]-c[j])**2))
                if dist < r[i] + r[j] - 1e-12:
                    return False
        return True

    # Iteratively reduce radii if invalid
    # But we want to maximize sum.
    # If the optimizer result is slightly invalid due to numerical precision or penalty softness,
    # we can scale radii down by (1 - epsilon).
    
    if not check_valid(centers_opt, radii_opt):
        # Find max overlap/boundary violation
        max_viol = 0
        for i in range(n):
            # Boundary
            for k in range(2):
                if centers_opt[i, k] < radii_opt[i]:
                    max_viol = max(max_viol, radii_opt[i] - centers_opt[i, k])
                if centers_opt[i, k] > 1 - radii_opt[i]:
                    max_viol = max(max_viol, centers_opt[i, k] + radii_opt[i] - 1)
            
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                if dist < radii_opt[i] + radii_opt[j]:
                    max_viol = max(max_viol, (radii_opt[i] + radii_opt[j]) - dist)
        
        if max_viol > 0:
            # Scale radii down
            # We need r_new + r_new <= dist => 2 r_new <= dist (if equal)
            # Generally, we can scale all radii by factor alpha < 1
            # But this reduces sum.
            # Better to just fix specific violations.
            # However, scaling is safe.
            # Find alpha such that valid.
            # Conservative: reduce by max_viol * 2?
            # Actually, if we scale radii by alpha, new sum of radii is alpha * old.
            # Overlap condition: dist >= alpha(r_i + r_j).
            # Boundary: x >= alpha r_i => r_i <= x/alpha.
            # This is tricky.
            
            # Let's just reduce radii slightly.
            radii_opt *= 0.999 # Very small reduction
            # And repeat check?
            # Or just use a loop to find a valid scaling factor.
            
            # Binary search for scaling factor s in [0, 1]
            # Such that s * radii_opt is valid.
            low, high = 0.0, 1.0
            for _ in range(20):
                mid = (low + high) / 2
                test_radii = radii_opt * mid
                if check_valid(centers_opt, test_radii):
                    low = mid
                else:
                    high = mid
            
            radii_opt = radii_opt * low

    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
