# sol_000150 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=eca47248 sum of radii=2.332998 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    num_trials = 5
    
    best_valid_sum = -1.0
    best_centers = None
    best_radii = None

    # Helper to compute valid radii and sum for a given set of centers
    def compute_valid_radii(centers):
        """
        Given centers, compute the largest possible valid radii for each circle
        such that they don't overlap and stay inside the square.
        Returns (radii, sum_radii).
        """
        r = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            dist_to_wall = min(centers[i, 0], 1 - centers[i, 0], 
                               centers[i, 1], 1 - centers[i, 1])
            
            # Distance to other centers (halved)
            dist_to_others = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < dist_to_others:
                    dist_to_others = d
            
            # Max radius is limited by the nearest obstacle (wall or another circle)
            # For circle-circle constraint: r_i + r_j <= d_ij. 
            # Here we assume we want to find a valid assignment. 
            # A safe conservative estimate for r_i given fixed centers and unknown r_j 
            # is not strictly defined without solving a system, but for the post-processing 
            # of an optimization that pushed radii up, we can check consistency.
            # However, the optimization variables included r.
            # To ensure strict validity: r_i <= dist_to_wall and r_i + r_j <= dist_ij.
            # If we fix centers, the maximum r_i is constrained by neighbors.
            # But neighbors also have radii. 
            # Actually, if the optimization found a configuration with radii R, 
            # and it's slightly invalid, we just need to reduce radii to make it valid.
            # The most robust way is to set r_i = min(dist_to_wall, min_j(dist_to_others/2)).
            # This assumes r_j could be as large as dist_to_others/2? No.
            # If r_j is small, r_i can be larger.
            # But we don't know r_j. 
            # However, if we just take the radii from the optimizer and clamp them?
            # Let's stick to the optimizer's radii but ensure constraints are met.
            # But for this helper, let's just return a safe valid radius based on geometry alone?
            # No, that assumes equal radii or worst case.
            pass 
        
        # Actually, let's just use the radii returned by optimizer but clamp them to satisfy constraints.
        # But checking constraints requires knowing all r.
        # Let's rely on the optimizer's r, but fix violations by shrinking.
        return r

    # Define the objective function for the optimizer
    def objective(vars):
        # vars layout: [x1, y1, ..., x26, y26, r1, ..., r26]
        # Length: 26*2 + 26 = 78
        
        centers = vars[:52].reshape(26, 2)
        radii = vars[52:]
        
        # Primary objective: maximize sum of radii => minimize -sum
        score = -np.sum(radii)
        
        penalty = 0.0
        penalty_weight = 5000.0 # High weight to enforce constraints
        
        # 1. Boundary constraints
        # r_i <= x_i  => r_i - x_i <= 0
        # r_i <= 1 - x_i => r_i + x_i - 1 <= 0
        # same for y
        
        # r - x
        violation = radii - centers[:, 0]
        penalty += np.sum(np.maximum(0, violation)**2)
        
        # r + x - 1
        violation = radii + centers[:, 0] - 1
        penalty += np.sum(np.maximum(0, violation)**2)
        
        # r - y
        violation = radii - centers[:, 1]
        penalty += np.sum(np.maximum(0, violation)**2)
        
        # r + y - 1
        violation = radii + centers[:, 1] - 1
        penalty += np.sum(np.maximum(0, violation)**2)
        
        # 2. Overlap constraints
        # (r_i + r_j)^2 <= ||c_i - c_j||^2
        # We penalize if (r_i + r_j) > distance
        
        # Vectorized calculation for pairs
        # c_diff: (26, 26, 2)
        c_diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(c_diff**2, axis=2) # (26, 26)
        
        # r_sum: (26, 26)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # We only care about i < j, but squaring the matrix is symmetric.
        # Violation amount: max(0, r_sum - sqrt(dists_sq))
        # To avoid sqrt singularity at 0, we can work with squares if careful, 
        # but sqrt is stable enough here.
        dists = np.sqrt(dists_sq)
        
        # Mask out diagonal (self-distance is 0)
        dists[np.arange(n), np.arange(n)] = np.inf
        
        overlap = r_sum - dists
        overlap[overlap < 0] = 0
        
        # Sum of squared violations
        penalty += np.sum(overlap**2)
        
        return score + penalty_weight * penalty

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius)
    bounds = []
    for _ in range(26):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Trial 1: Grid Initialization (5x5 + 1)
    # 25 circles in 5x5 grid, 1 in the center of a gap
    centers_grid = []
    for r in range(5):
        for c in range(5):
            centers_grid.append([0.1 + r*0.2, 0.1 + c*0.2])
    # Add 26th circle at a gap, e.g., (0.2, 0.2)
    centers_grid.append([0.2, 0.2])
    centers_grid = np.array(centers_grid)
    
    # Initial radii: small enough to not overlap immediately, but significant
    # 5x5 grid with spacing 0.2 -> diameter 0.2 -> radius 0.1 fits perfectly.
    # But we have 26 circles, so 0.1 might be too tight. Start at 0.08.
    radii_grid = np.full(n, 0.08)
    
    x0_grid = np.concatenate([centers_grid.flatten(), radii_grid])
    
    # Trial 2: Random Initialization
    np.random.seed(42)
    centers_rand = np.random.rand(n, 2) * 0.8 + 0.1 # Keep away from edges initially
    radii_rand = np.full(n, 0.05)
    x0_rand = np.concatenate([centers_rand.flatten(), radii_rand])

    # Trial 3: Hexagonal-ish Initialization
    # Try to fit 26 circles in a staggered grid
    centers_hex = []
    # 4 rows of 7? 4*7=28 (too many). 
    # 6, 5, 6, 5, 4 = 26.
    rows = [6, 5, 6, 5, 4]
    y_coord = 0.1 # Start y
    dy = 0.15 # Vertical spacing
    for count in rows:
        # Center the row
        # Width available 1.0. 
        # If count circles, spacing dx = 1.0 / (count + 1) ?
        # Or just distribute evenly.
        # Let's distribute evenly in [0.05, 0.95]
        xs = np.linspace(0.1, 0.9, count)
        for x in xs:
            centers_hex.append([x, y_coord])
        y_coord += dy
    # If we didn't get 26, pad or trim (should be 26)
    # 6+5+6+5+4 = 26. Correct.
    centers_hex = np.array(centers_hex)
    radii_hex = np.full(n, 0.06)
    x0_hex = np.concatenate([centers_hex.flatten(), radii_hex])

    trials = [x0_grid, x0_rand, x0_hex]
    
    # Run optimization
    for i, x0 in enumerate(trials):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-9})
            
            # Extract solution
            c_opt = res.x[:52].reshape(26, 2)
            r_opt = res.x[52:]
            
            # Post-processing: Enforce strict validity
            # The optimizer might have small overlaps. 
            # We adjust radii to be strictly valid.
            # We can iteratively shrink radii that violate constraints.
            
            # Simple validity enforcement:
            # For each circle, r_i cannot exceed dist to wall.
            # Also r_i + r_j <= dist(c_i, c_j).
            # This is a system. A simple heuristic: 
            # r_i = min(r_i, dist_to_wall_i, min_j( (dist_ij - r_j)/2 )) ? 
            # This requires r_j.
            
            # Better: Just take the radii and if invalid, reduce them.
            # But we need a valid configuration.
            # Let's use a simple "clamp" logic.
            # 1. Clamp to walls.
            for k in range(n):
                c = c_opt[k]
                r = r_opt[k]
                max_r_wall = min(c[0], 1-c[0], c[1], 1-c[1])
                if r > max_r_wall:
                    r_opt[k] = max_r_wall
            
            # 2. Resolve overlaps by shrinking the smaller circle? 
            # Or just ensure r_i + r_j <= dist.
            # Iterate a few times to settle.
            for _ in range(50):
                changed = False
                dists = np.sqrt(np.sum((c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :])**2, axis=2))
                for k in range(n):
                    for m in range(k+1, n):
                        d = dists[k, m]
                        sum_r = r_opt[k] + r_opt[m]
                        if sum_r > d:
                            # Overlap. Reduce radii.
                            # Reduce both proportionally or reduce the one that hurts sum less?
                            # To maximize sum, we want to reduce as little as possible.
                            # Constraint: r_k + r_m <= d.
                            # Excess = sum_r - d.
                            # We need to reduce sum_r by excess.
                            # Split excess between k and m.
                            excess = sum_r - d
                            # Reduce both by half
                            reduce = excess / 2.0
                            r_opt[k] -= reduce
                            r_opt[m] -= reduce
                            if r_opt[k] < 0: r_opt[k] = 0
                            if r_opt[m] < 0: r_opt[m] = 0
                            changed = True
                
                # Also check walls again after shrinking
                for k in range(n):
                    c = c_opt[k]
                    max_r_wall = min(c[0], 1-c[0], c[1], 1-c[1])
                    if r_opt[k] > max_r_wall:
                        r_opt[k] = max_r_wall
                        changed = True
                
                if not changed:
                    break
            
            current_sum = np.sum(r_opt)
            if current_sum > best_valid_sum:
                best_valid_sum = current_sum
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
                
        except Exception:
            continue

    # Final Validation Check (mental check of logic)
    # The post-processing loop ensures r_i >= 0 and r_i <= wall_dist and r_i+r_j <= dist.
    # It might not be globally optimal sum, but it's valid.
    # However, the loop might reduce sum significantly if optimizer failed.
    # The penalty method with high weight should yield a valid config close to optimum.
    
    return best_centers, best_radii, best_valid_sum
