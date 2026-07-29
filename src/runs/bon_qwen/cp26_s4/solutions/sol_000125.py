# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 40ff4175) state=463805e4 sum of radii=0.076983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Strategy: 
    # 1. Initialize centers in a hexagonal-like pattern.
    # 2. Use an optimization routine to maximize the sum of radii.
    #    We will optimize variables [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26].
    #    However, optimizing 78 variables with many constraints is hard.
    #    Instead, we can fix radii to be equal initially or optimize positions for a target radius.
    #    A robust approach: Maximize the minimum distance between circles and boundaries.
    #    If we find positions P_i that maximize min_dist, then r = min_dist / 2 is a valid radius for all.
    #    But we want to maximize sum of radii. Allowing different radii might help.
    #    
    #    Let's try a force-directed approach which is often effective for packing.
    #    We simulate repulsive forces.
    
    # --- Initialization ---
    # Create a grid of points, then perturb to hexagonal
    # Or simply start with a dense grid and let optimization spread them out.
    
    # Let's try a 6x5 grid pattern (30 points) and remove 4, or just pick 26.
    # Better: Hexagonal lattice.
    
    centers = np.zeros((n, 2))
    
    # Generate points on a hexagonal lattice
    # We want to fill [0,1]x[0,1].
    # Approximate radius r ~ 0.1. Spacing ~ 0.2.
    # Let's generate a list of candidate points with spacing 0.15
    candidates = []
    spacing = 0.15
    y = spacing
    row = 0
    while y < 1.0:
        x_start = spacing if row % 2 == 0 else spacing * 0.5 # Offset for hex
        x = x_start
        while x < 1.0:
            candidates.append([x, y])
            x += spacing
        y += spacing * math.sqrt(3) / 2 # Vertical spacing for hex
        row += 1
    
    # If we have more candidates, select the first 26 or random subset
    # If fewer, add some random points
    candidates = np.array(candidates)
    if len(candidates) >= n:
        # Select a subset that is well distributed
        # Simple strategy: take every k-th point or just first n
        # To avoid clustering, maybe shuffle?
        # But for deterministic result, let's take first n
        # However, first n might be clustered in one corner.
        # Let's pick points that maximize spread?
        # Simple greedy: pick point furthest from existing set
        selected_indices = []
        if len(candidates) > 0:
            # Pick center-most or random
            idx = len(candidates) // 2
            selected_indices.append(idx)
            while len(selected_indices) < n and len(selected_indices) < len(candidates):
                best_dist = -1
                best_idx = -1
                for i in range(len(candidates)):
                    if i in selected_indices:
                        continue
                    # Min distance to selected
                    min_d = float('inf')
                    for sel in selected_indices:
                        d = np.linalg.norm(candidates[i] - candidates[sel])
                        if d < min_d:
                            min_d = d
                    if min_d > best_dist:
                        best_dist = min_d
                        best_idx = i
                if best_idx != -1:
                    selected_indices.append(best_idx)
        
        centers = candidates[selected_indices[:n]].copy()
    else:
        # If not enough, fill remaining with random
        existing = candidates.tolist()
        while len(existing) < n:
            existing.append([np.random.rand(), np.random.rand()])
        centers = np.array(existing[:n])

    # Initial radii
    # Estimate radius based on density
    # Area 1, 26 circles -> area/circle ~ 0.038 -> r ~ sqrt(0.038/pi) ~ 0.11
    # But packing efficiency < 1. Let's start with 0.05
    radii = np.ones(n) * 0.05
    
    # --- Optimization ---
    # We will use a simple iterative expansion and relaxation.
    # 1. Try to increase radii.
    # 2. If overlap, push centers apart.
    # 3. Repeat.
    
    # This is a "repulsive force" simulation.
    
    num_iterations = 1000
    # We want to maximize sum of radii.
    # Let's maintain a "target" radius or just let them grow.
    # To prevent them from growing infinitely (they are bounded by square),
    # we can define a potential energy.
    
    # However, a simpler heuristic that works well:
    # Run a solver that maximizes min(clearance).
    # Clearance of circle i = min(x_i, 1-x_i, y_i, 1-y_i, min_j dist(i,j)/2)
    # Then r_i = clearance_i.
    # Maximize sum of clearances? No, that's hard.
    # Maximize min clearance (which gives equal radii).
    
    # Let's try to maximize equal radius r.
    # Variables: 26*(x,y). 
    # Objective: maximize r.
    # Constraints: dist >= 2r, boundary >= r.
    # This is equivalent to finding the "maximin" configuration.
    
    # Let's implement a function to compute the max possible equal radius for a given set of centers.
    # r(centers) = min( min_i dist_to_boundary(i), min_{i,j} dist(i,j)/2 )
    # We want to maximize this r.
    
    # We can use scipy.optimize to maximize r.
    # But r is determined by centers.
    # So we maximize f(centers) = min( ... )
    # This is a maximin problem.
    # f is non-smooth.
    # We can smooth it or use a solver that handles min.
    # Or just use the force simulation.
    
    # Force Simulation Implementation
    centers = centers.copy() # local copy
    radii = np.ones(n) * 0.05 # Start small
    
    # We will iteratively increase radii and resolve conflicts.
    # This is similar to "expanding circles".
    
    growth_step = 0.0005
    max_r = 0.0
    
    # To help escape local minima, we can jitter.
    
    for step in range(2000):
        # Increase radii slightly
        current_r = radii[0] # Assuming equal radii for simplicity in this loop
        # Actually, let's just track a global radius r
        r = current_r
        r += growth_step
        
        # Check if this r is feasible with current centers
        # If not, move centers to make it feasible.
        
        # Instead of checking feasibility, let's apply forces.
        # Force = repulsion if dist < 2*r.
        
        # But we want to increase r.
        # Let's say r is the variable we want to increase.
        # We keep r fixed for a few iterations to let centers settle, then increase.
        
        # Apply forces
        force = np.zeros_like(centers)
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r and dist > 1e-9:
                    # Repulsive force
                    # Magnitude proportional to overlap
                    overlap = 2 * r - dist
                    direction = diff / dist
                    # Spring force
                    f = overlap * 10.0 # stiffness
                    force[i] += f * direction
                    force[j] -= f * direction
                elif dist < 1e-9:
                    # Same position, push randomly
                    rand_dir = np.random.randn(2)
                    force[i] += rand_dir
                    force[j] -= rand_dir
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            if x < r:
                force[i, 0] += (r - x) * 10.0
            if x > 1 - r:
                force[i, 0] -= (x - (1 - r)) * 10.0
            if y < r:
                force[i, 1] += (r - y) * 10.0
            if y > 1 - r:
                force[i, 1] -= (y - (1 - r)) * 10.0
        
        # Apply forces
        # Adaptive step size for movement
        # If forces are large, move more?
        # But we must stay within [0,1].
        
        # Simple Euler step
        move_step = 0.5 # damping
        centers += move_step * force
        
        # Clip centers to valid range [r, 1-r] is hard because r changes.
        # But centers must be in [0,1] at least.
        centers = np.clip(centers, 0.0, 1.0)
        
        # Check if valid with radius r
        # If valid, we can try to increase r more aggressively next time?
        # If not valid, maybe decrease growth_step?
        
        # Validation check (approximate)
        valid = True
        # Check boundaries
        for i in range(n):
            if centers[i, 0] < r or centers[i, 0] > 1 - r or \
               centers[i, 1] < r or centers[i, 1] > 1 - r:
                valid = False
                break
        if valid:
            # Check overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < 2 * r - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Increase growth rate slightly
            growth_step = min(growth_step * 1.01, 0.002)
            radii[:] = r
        else:
            # Overlap or boundary violation.
            # Reduce growth rate to allow settling
            growth_step = max(growth_step * 0.99, 1e-6)
            # Also, maybe we should decrease r slightly to let centers move?
            # But here we are just applying forces based on target r.
            # If forces are strong enough, centers will move.
            pass

    # After simulation, we have a set of centers and a radius r.
    # But radii might not be uniform in optimal solution?
    # Let's assume equal radii is a good approximation.
    # The final r in the loop is the radius we attempted.
    # We need to find the actual max valid radius for these centers.
    
    # Calculate the max possible radius for the current centers
    min_dist = 1.0
    for i in range(n):
        # Distance to boundaries
        d_bound = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        if d_bound < min_dist:
            min_dist = d_bound
            
        for j in range(i + 1, n):
            d_pair = np.linalg.norm(centers[i] - centers[j])
            if d_pair < min_dist:
                min_dist = d_pair
    
    final_r = min_dist / 2.0
    
    # Set all radii to final_r
    radii = np.ones(n) * final_r
    
    # However, we might be able to improve by allowing different radii.
    # But for 26 circles, equal is likely very good.
    # Let's try to perturb and optimize sum of radii with a solver.
    
    # We have a valid packing with radius final_r.
    # Let's try to optimize centers to maximize sum of radii.
    # But with equal radii constraint, maximizing sum is maximizing r.
    
    # Let's run a local optimization on the centers to maximize min_dist.
    # We can use scipy.optimize.minimize with a smooth approximation of min.
    # Or just use the force simulation result which is usually quite good.
    
    # To be safe, let's run a few more steps of optimization specifically for min_dist.
    # We want to maximize min_dist.
    # This is equivalent to minimizing -min_dist.
    # We can use a penalty method on -min_dist?
    # Actually, we can just run the force simulation for longer with a fixed r?
    # No, we want to find the best r.
    
    # Let's try a small perturbation optimization using scipy.
    # Objective: Maximize r such that constraints hold.
    # We can parameterize: variables are centers. r is computed.
    # Maximize r(centers).
    
    # Let's define a function that returns the max valid r for given centers.
    def get_max_r(centers_flat):
        c = centers_flat.reshape(-1, 2)
        r = 1.0
        # Boundary
        for i in range(n):
            r = min(r, c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1])
        # Pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                r = min(r, d / 2.0)
        return r

    # We want to maximize get_max_r.
    # Use Nelder-Mead or Powell since gradient is not smooth (min function).
    # But Nelder-Mead is derivative free.
    
    from scipy.optimize import minimize
    
    # Current centers
    x0 = centers.flatten()
    
    # Minimize negative of max_r
    # But get_max_r is expensive (O(N^2)). N=26 is small, so it's fine.
    
    # To speed up, we can use a simple hill climbing or just rely on simulation.
    # Let's try a few random restarts of the simulation to find a better packing.
    # But we are in a function, time might be limited? 
    # The prompt doesn't specify time limit, but "generously rewarded" suggests quality matters.
    
    # Let's do a quick local optimization using scipy 'Nelder-Mead'
    # It might take a bit but 26 circles is small.
    
    # We need to clip centers during optimization to stay in [0,1].
    # Nelder-Mead doesn't support bounds easily.
    # But we can penalize out of bounds.
    
    def objective(centers_flat):
        c = centers_flat.reshape(-1, 2)
        # Penalty for out of bounds
        penalty = 0.0
        for i in range(n):
            if c[i, 0] < 0: penalty += 100 * c[i, 0]**2
            if c[i, 0] > 1: penalty += 100 * (c[i, 0] - 1)**2
            if c[i, 1] < 0: penalty += 100 * c[i, 1]**2
            if c[i, 1] > 1: penalty += 100 * (c[i, 1] - 1)**2
        
        if penalty > 0:
            return 1000 + penalty # High cost
            
        r = 1.0
        for i in range(n):
            val = min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1])
            if val < r: r = val
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                val = d / 2.0
                if val < r: r = val
        return -r # Maximize r -> minimize -r

    # Run optimization
    # Reshape centers
    x0 = centers.flatten()
    
    # Use a simple optimization. 
    # Since the function is non-smooth, Nelder-Mead is good.
    # But it might be slow.
    # Let's try SLSQP with a smooth approximation?
    # Or just run Nelder-Mead with a limit.
    
    # Actually, the force simulation already did a lot of work.
    # The result should be close to optimal.
    # Let's just return the result from simulation.
    # But wait, the simulation maintained equal radii.
    # Is it possible to have unequal radii with higher sum?
    # Maybe. But for N=26, the difference is likely small.
    # The simulation with equal radii should yield r ~ 0.101 or so.
    
    # Let's verify the radius we found.
    # If r is around 0.1, sum is 2.6.
    # Target 2.636 implies r ~ 0.1014.
    # My simulation should find this if initialized well.
    
    # To ensure we get a high quality solution, let's try to refine the equal radius packing.
    # We can use the 'minimize' function to maximize r.
    
    # Let's wrap the objective and run minimize.
    # We need to handle the non-smoothness.
    # A common trick is to use the "soft min" or just rely on the solver handling the flat regions.
    # Nelder-Mead handles non-smooth well.
    
    # Let's run Nelder-Mead for a few iterations.
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.fun < 0: # fun is -r, so if negative, r > 0
            centers_opt = res.x.reshape(-1, 2)
            # Recalculate r
            r_opt = get_max_r(res.x)
            # Check validity
            # Clip just in case
            centers_opt = np.clip(centers_opt, 0, 1)
            # Recalculate r with clipped
            r_opt = get_max_r(centers_opt.flatten())
            
            # If we found a better r, use it
            if r_opt > final_r - 1e-9:
                centers = centers_opt
                final_r = r_opt
    except Exception:
        pass # Fallback to simulation result

    radii = np.ones(n) * final_r
    
    # Double check validity with the provided validator logic (approximate)
    # Just to be sure, we can clamp radii slightly if needed, but get_max_r ensures validity.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
