import numpy as np
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative
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

def get_hexagonal_initialization(n):
    """
    Creates a hexagonal packing initialization for n circles.
    """
    centers = []
    radii = []
    
    # Start with a radius estimate. For 26 circles, r ~ 0.1.
    # Hexagonal packing density is higher, so maybe r ~ 0.101.
    r = 0.095 
    
    # We will place circles in a hexagonal grid
    # Row 0: y = r, x = r, r+2r, r+4r...
    # Row 1: y = r + r*sqrt(3), x = 2r, 4r... (shifted by r)
    
    rows = []
    row_idx = 0
    while len(centers) < n:
        y = r + row_idx * (r * math.sqrt(3))
        # Check if row fits in height
        if y + r > 1.0:
            # If row doesn't fit, try to adjust or stop. 
            # For initialization, we just place them and let optimizer fix.
            pass 
            
        # Determine x offset
        if row_idx % 2 == 0:
            x_start = r
        else:
            x_start = 2 * r
            
        current_row_centers = []
        x = x_start
        while x + r <= 1.0 + 1e-9:
            current_row_centers.append((x, y))
            x += 2 * r
        rows.append(current_row_centers)
        
        for c in current_row_centers:
            if len(centers) < n:
                centers.append(c)
            else:
                break
        row_idx += 1
        
    # Trim or pad to n
    centers = centers[:n]
    # If not enough, just fill with random points
    while len(centers) < n:
        centers.append((np.random.uniform(0, 1), np.random.uniform(0, 1)))
        
    radii = np.full(n, r)
    return np.array(centers), radii

def objective_func(params, n):
    """
    Objective function for optimization.
    params: array of size [n, 3] -> [x, y, r]
    Returns negative sum of radii (to minimize)
    """
    # Reshape
    centers = params[:, :2]
    radii = params[:, 2]
    
    # Enforce radius > 0
    radii = np.maximum(radii, 1e-6)
    
    # Calculate sum
    current_sum = np.sum(radii)
    
    # Return negative sum to maximize
    return -current_sum

def constraints_func(params, n):
    """
    Returns constraint violations.
    We use a soft penalty or just rely on optimizer bounds?
    scipy.optimize.minimize with 'SLSQP' supports constraints.
    But for circle packing, constraints are complex.
    A simpler approach for global optimization is to use a penalty method 
    or just a randomized search with local optimization.
    """
    pass

import scipy.optimize

def run_packing():
    n = 26
    # Number of variables: n circles * (x, y, r) = 78 variables.
    # This is quite high for global optimization.
    # Strategy: Fix radii to be equal first? Or optimize equal radii?
    # Let's try optimizing equal radii configuration first to find a good layout,
    # then relax radii.
    
    # Actually, let's just run a local optimizer from a good hexagonal start.
    # We can treat r as variable.
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Try multiple random restarts or variations of hexagonal packing
    for seed in range(20):
        np.random.seed(seed)
        
        # Initial guess
        # Let's generate a perturbed hexagonal grid
        init_centers, init_radii = get_hexagonal_initialization(n)
        
        # Perturb centers slightly
        perturbation = np.random.uniform(-0.02, 0.02, init_centers.shape)
        centers_perturbed = init_centers + perturbation
        # Clamp centers to [0, 1] roughly
        centers_perturbed = np.clip(centers_perturbed, 0.05, 0.95)
        
        # Initial radii can be slightly different
        radii_perturbed = init_radii + np.random.uniform(-0.01, 0.01, n)
        radii_perturbed = np.maximum(radii_perturbed, 0.05)
        
        # Flatten for optimizer
        x0 = np.hstack([centers_perturbed.flatten(), radii_perturbed.flatten()])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # We need to handle constraints. 
        # Since we want to maximize sum of radii, let's use a penalty method inside the objective.
        # But standard optimizers don't handle "non-overlap" well as soft constraints without tuning.
        
        # Alternative: Use scipy.optimize.minimize with 'SLSQP' and explicit constraints?
        # Too many constraints (N*(N-1)/2 ~ 325 constraints).
        
        # Better approach: Use a custom gradient descent or simulated annealing?
        # Or just 'Nelder-Mead' with a penalty function.
        
        def penalized_objective(p):
            # p is flat array
            centers = p[:2*n].reshape((n, 2))
            radii = p[2*n:]
            
            # Penalty for constraints
            penalty = 0.0
            
            # Boundary penalties
            # If circle i is outside, penalize
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if r < 0: r = 0
                
                # Distance to boundaries
                dist_left = x - r
                dist_right = (1 - x) - r
                dist_down = y - r
                dist_up = (1 - y) - r
                
                min_dist = min(dist_left, dist_right, dist_down, dist_up)
                if min_dist < 0:
                    penalty += 10000 * (min_dist ** 2)
            
            # Overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    sum_r = radii[i] + radii[j]
                    if dist < sum_r:
                        violation = sum_r - dist
                        penalty += 10000 * (violation ** 2)
            
            # We want to maximize sum of radii, so minimize -sum - penalty?
            # Actually, penalty should make infeasible solutions very bad.
            # Objective: -sum(radii) + penalty
            return -np.sum(radii) + penalty

        # Nelder-Mead is robust but slow. 
        # Let's try Powell or Nelder-Mead.
        try:
            res = scipy.optimize.minimize(penalized_objective, x0, method='Nelder-Mead', 
                                          options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Extract result
            res_centers = res.x[:2*n].reshape((n, 2))
            res_radii = res.x[2*n:]
            
            # Validate
            if validate_packing(res_centers, res_radii):
                s = np.sum(res_radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = res_centers.copy()
                    best_radii = res_radii.copy()
        except Exception as e:
            pass

    # If we didn't find a good solution, fallback to equal radius hexagonal?
    if best_sum < 2.5:
        # Simple 5x5 grid + 1
        # But we need to improve.
        # Let's try to optimize equal radii specifically with fewer variables?
        # Variables: 2*n coordinates. Radius r is scalar.
        # But SLSQP with 52 vars and ~325 constraints is heavy.
        
        # Let's stick to the result from the loop. 
        # 20 iterations of Nelder-Mead should find something decent.
        
        # One more trick: scale the solution to make it tighter if valid.
        # If valid, we can try to expand radii uniformly?
        # But radii might be unequal.
        
        # Let's just return the best found.
        pass

    return best_centers, best_radii, best_sum

# To make it more robust, let's refine the optimization strategy.
# Nelder-Mead might get stuck. 
# Let's try a simple iterative expansion.
# Start with small circles, place them, then try to expand.

def run_packing_refined():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Try a few seeds
    for seed in range(50):
        np.random.seed(seed)
        
        # Initial placement: random points
        centers = np.random.uniform(0.05, 0.95, (n, 2))
        radii = np.full(n, 0.05) # Start small
        
        # Iterative expansion
        # Simple approach: randomly pick a circle, increase radius, check constraints, rollback if fail.
        # Repeat many times.
        
        for step in range(5000):
            idx = np.random.randint(n)
            old_r = radii[idx]
            # Try to increase radius
            new_r = old_r + 0.001 + np.random.random() * 0.005
            
            radii[idx] = new_r
            
            # Check constraints for this circle
            valid = True
            
            # Boundary
            x, y = centers[idx]
            if x - new_r < 0 or x + new_r > 1 or y - new_r < 0 or y + new_r > 1:
                valid = False
            
            # Overlap
            if valid:
                for j in range(n):
                    if i != j: # Wait, need loop variable
                        pass
                # Check overlap with all others
                for j in range(n):
                    if idx == j: continue
                    dist = np.sqrt(np.sum((centers[idx] - centers[j]) ** 2))
                    if dist < new_r + radii[j]:
                        valid = False
                        break
            
            if not valid:
                radii[idx] = old_r # Rollback
            else:
                # Also try to move center to allow more expansion?
                # Maybe perturb center slightly?
                pass
        
        # This "grow" heuristic is slow and might not reach optimal.
        # Better to use the optimizer with penalty.
        pass

    # Let's return to the optimizer approach but maybe tune it.
    # Or just use the provided code structure with the first strategy.
    # The first strategy (Nelder-Mead with penalty) is good.
    # Let's execute that.
    
    # Re-defining run_packing to be the final function
    return run_packing()

# Actually, I need to provide the code for run_packing.
# I will combine the ideas.

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Optimization parameters
    penalty_weight = 10000.0
    
    def penalized_objective(p):
        centers = p[:2*n].reshape((n, 2))
        radii = p[2*n:]
        
        # Soften negative radii
        radii = np.maximum(radii, 1e-9)
        
        # Boundary penalty
        # dist to boundary
        # x in [r, 1-r] => r <= x and r <= 1-x => r <= min(x, 1-x)
        # Violation: if r > min(x, 1-x)
        
        penalty = 0.0
        
        # Vectorized boundary check
        # min_dist to wall
        min_x = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
        min_y = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        wall_dist = np.minimum(min_x, min_y)
        
        violations = radii - wall_dist
        # Only positive violations count
        violations = np.maximum(violations, 0)
        penalty += penalty_weight * np.sum(violations ** 2)
        
        # Overlap penalty
        # Compute all pairwise distances
        # Centers shape (n, 2)
        # Diff shape (n, n, 2)
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        
        # Sum of radii
        sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # We only care about i < j, but matrix is symmetric.
        # Mask upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Distances
        dists = np.sqrt(dists_sq)
        
        # Violations: sum_radii - dists > 0
        overlap_violations = sum_radii - dists
        # Clip negative (non-overlap) to 0
        overlap_violations = np.maximum(overlap_violations, 0)
        
        # Sum squared violations
        penalty += penalty_weight * np.sum(overlap_violations[mask] ** 2)
        
        # Objective: maximize sum of radii => minimize -sum
        return -np.sum(radii) + penalty

    # Run multiple restarts
    for seed in range(30):
        np.random.seed(seed)
        
        # Initialization: Hexagonal
        centers_init, radii_init = get_hexagonal_initialization(n)
        
        # Perturb
        centers_init += np.random.normal(0, 0.01, centers_init.shape)
        centers_init = np.clip(centers_init, 0.05, 0.95)
        radii_init += np.random.normal(0, 0.005, n)
        radii_init = np.maximum(radii_init, 0.05)
        
        x0 = np.hstack([centers_init.flatten(), radii_init.flatten()])
        
        # Bounds
        bnds = []
        for _ in range(n):
            bnds.append([0.0, 1.0]) # x
            bnds.append([0.0, 1.0]) # y
            bnds.append([0.0, 0.5]) # r
            
        # Use Powell or Nelder-Mead
        try:
            res = scipy.optimize.minimize(penalized_objective, x0, method='Nelder-Mead',
                                          options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            centers_cand = res.x[:2*n].reshape((n, 2))
            radii_cand = res.x[2*n:]
            
            # Hard clip radii to be positive
            radii_cand = np.maximum(radii_cand, 1e-9)
            
            # Validate
            if validate_packing(centers_cand, radii_cand):
                s = np.sum(radii_cand)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers_cand.copy()
                    best_radii = radii_cand.copy()
        except:
            pass

    # Fallback if nothing found (unlikely)
    if best_centers is None:
        centers_fallback, radii_fallback = get_hexagonal_initialization(n)
        return centers_fallback, radii_fallback, np.sum(radii_fallback)

    return best_centers, best_radii, best_sum