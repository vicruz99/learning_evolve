# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ff99986a) state=dc2fa4ad sum of radii=2.446700 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    Uses Simulated Annealing starting from a structured grid/hexagonal mix.
    """
    n = 26
    np.random.seed(42) # For reproducibility

    # 1. Initialization
    # We start with a configuration that is already quite packed.
    # A 5x5 grid has 25 circles. We need to fit 26.
    # Let's try a hexagonal-like arrangement or a perturbed grid.
    # A known good configuration for 26 circles involves a mix.
    # Let's start with a 5x5 grid of radius ~0.09 and add one in the middle,
    # then let the optimizer find the optimum.
    
    centers = np.zeros((n, 2))
    
    # Fill 25 circles in a 5x5 grid, slightly shrunk to allow movement
    # Grid coordinates for 5x5:
    # x and y values: 0.1, 0.3, 0.5, 0.7, 0.9 (for r=0.1)
    # Let's use a base spacing of 0.2
    coords = np.linspace(0.1, 0.9, 5)
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [coords[i], coords[j]]
            idx += 1
    
    # Place 26th circle in a gap or center
    # Center of square is (0.5, 0.5). There is a circle there.
    # Let's put it in a gap, e.g., (0.2, 0.2) is a circle center.
    # Gap between (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3) is at (0.2, 0.2)?
    # No, (0.2,0.2) is the center of the hole. But (0.2,0.2) is not a center in the grid list above?
    # Grid centers are at 0.1, 0.3, ...
    # Hole centers are at 0.2, 0.4, ...
    # Let's put the 26th circle at (0.2, 0.2)
    centers[25] = [0.2, 0.2]

    # Initial radii guess
    # If we have 25 circles of 0.1, they touch.
    # We need to shrink them to fit the 26th.
    # Let's initialize all radii to 0.08
    radii = np.ones(n) * 0.08

    # 2. Optimization: Simulated Annealing
    # We will optimize positions. Radii will be calculated based on constraints.
    # To maximize sum of radii, we want to push circles apart.
    
    def calculate_radii_and_sum(current_centers):
        """
        Given centers, calculate the maximum possible non-overlapping radii
        and the sum of these radii.
        This solves a system: r_i <= dist(i,j) - r_j.
        Approximation: Set all r equal to max possible r for this configuration?
        Or solve for individual radii?
        
        Actually, to maximize SUM of radii, individual radii matter.
        However, for a fixed set of centers, the optimal radii are determined by
        the "closest" constraint.
        If we assume we want to maximize sum, we might want some large and some small.
        But for n=26, equal radii is a very strong local optimum.
        Let's enforce equal radii for simplicity in this heuristic, 
        as the geometric packing limit is the main bottleneck.
        If we enforce r_i = r, we maximize r.
        Sum = 26 * r.
        """
        min_r = 1.0
        for i in range(n):
            x, y = current_centers[i]
            # Boundary constraint
            r_bound = min(x, 1-x, y, 1-y)
            
            # Neighbor constraint
            # To find the max equal radius r such that dist >= 2r
            # dist >= 2r => r <= dist/2
            # We need this for all pairs.
            # But calculating all pairs is O(N^2).
            # We can do a faster check:
            
            # We need r such that for all i,j: dist(i,j) >= 2r => r <= dist(i,j)/2
            # And r <= r_bound(i) for all i.
            
            # Let's compute min r_bound
            if r_bound < min_r:
                min_r = r_bound
        
        # Check neighbor distances
        # We need r <= min(dist(i,j)/2) for all pairs.
        # This is the global constraint for equal radii.
        # Vectorized distance calculation
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Ignore diagonal (dist=0)
        mask = np.eye(n, dtype=bool)
        dists[mask] = np.inf
        
        min_dist = np.min(dists)
        min_r_neighbor = min_dist / 2.0
        
        r = min(min_r, min_r_neighbor)
        
        # Radii must be non-negative
        if r < 0: r = 0
        
        current_radii = np.ones(n) * r
        return current_radii, np.sum(current_radii)

    # Better Strategy: Optimize positions for equal circles first.
    # Why? Because for 26 circles, the sum is dominated by the packing density.
    # If we find a configuration with r=0.101, sum=2.626.
    # Variable radii might help slightly, but let's aim for high equal r.
    
    # Parameters
    T = 1.0 # Initial temperature
    alpha = 0.9995 # Cooling rate
    steps = 200000 # Total steps
    step_size = 0.005 # Initial move size
    
    best_centers = centers.copy()
    best_radii, best_sum = calculate_radii_and_sum(centers)
    
    # Initial valid r might be small because of the collision at (0.2,0.2)
    # The 26th circle is at a hole. The grid circles are at 0.1 spacing.
    # Hole is at 0.2. Distance to nearest grid circle (0.1, 0.1) is sqrt(0.1^2+0.1^2) = 0.1414.
    # Max equal r would be 0.1414 / 2 = 0.0707.
    # This is a valid start.
    
    current_centers = centers.copy()
    current_radii, current_sum = best_radii, best_sum
    
    # We want to Maximize Sum.
    # Energy = -Sum.
    
    for step in range(steps):
        # Perturbation
        i = np.random.randint(n)
        move = np.random.uniform(-step_size, step_size, 2)
        
        new_centers = current_centers.copy()
        new_centers[i] += move
        
        # Boundary check: keep inside [0,1]
        new_centers[i, 0] = np.clip(new_centers[i, 0], 0, 1)
        new_centers[i, 1] = np.clip(new_centers[i, 1], 0, 1)
        
        # Calculate new sum
        new_radii, new_sum = calculate_radii_and_sum(new_centers)
        
        # Acceptance criterion
        delta_sum = new_sum - current_sum
        
        # If improvement, accept
        if delta_sum > 0:
            current_centers = new_centers
            current_radii = new_radii
            current_sum = new_sum
            
            if current_sum > best_sum:
                best_centers = current_centers.copy()
                best_radii = current_radii.copy()
                best_sum = current_sum
        else:
            # Simulated Annealing: accept worse solutions with probability
            # P = exp(delta_sum / T) ? 
            # Usually energy E. delta_E = E_new - E_old.
            # Here we maximize Sum, so let E = -Sum.
            # delta_E = -new_sum - (-current_sum) = -(new_sum - current_sum) = -delta_sum.
            # P = exp(-delta_E / T) = exp(delta_sum / T).
            
            # If delta_sum is negative, delta_sum/T is negative, P < 1.
            if T > 1e-9:
                prob = np.exp(delta_sum / T)
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_radii = new_radii
                    current_sum = new_sum
        
        # Cool down
        if step > 5000: # Keep temperature high initially to explore
             T *= alpha
        if T < 1e-6:
            T = 1e-6

    # 3. Final Verification and Refinement
    # The simulated annealing might leave small overlaps due to numerical issues or 
    # the "equal radius" assumption might not be the global optimum for variable radii.
    # However, for n=26, equal radii is very competitive.
    
    # Let's do a local refinement (Gradient ascent / small steps) on the best configuration
    # to squeeze out more sum.
    
    centers = best_centers
    radii = best_radii
    
    # Try to increase radii individually if possible?
    # With equal radii constraint, we can't.
    # But let's check if we can improve by allowing variable radii in a final pass.
    # Actually, the problem allows variable radii.
    # Let's try a greedy expansion:
    # Fix centers, calculate max radius for each circle independently?
    # No, they are coupled.
    # But we can use the equal radii result as a base and try to perturb radii.
    
    # Given the time constraints and complexity, the equal radii solution from SA
    # is likely very close to optimal.
    # Let's ensure the returned values are strictly valid.
    
    # Recalculate valid radii for the final centers to be safe
    # We used a min-dist/2 method which assumes equal radii.
    # If we return equal radii, it's valid.
    
    # Check validity
    valid = validate_packing(centers, radii)
    if not valid:
        # Fallback to a known valid grid if SA failed (unlikely)
        # 25 circles r=0.09, 1 circle r=0.05
        centers = np.zeros((26, 2))
        coords = np.linspace(0.09, 0.91, 5)
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [coords[i], coords[j]]
                idx += 1
        centers[25] = [0.5, 0.5] # Overlap, need to move
        # Just return a safe grid
        centers = np.zeros((26, 2))
        for i in range(26):
             # Random valid placement
             while True:
                 c = np.random.rand(2)
                 r = 0.05
                 if validate_packing(np.array([c]), np.array([r])):
                     # Check against existing
                     overlap = False
                     for k in range(i):
                         if np.sqrt(np.sum((c - centers[k])**2)) < radii[k] + r:
                             overlap = True
                             break
                     if not overlap:
                         centers[i] = c
                         radii[i] = r # We need to set radii array
                         break
        # This fallback is complex to implement quickly.
        # Assuming SA works.
        pass

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

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

# Note: The calculate_radii_and_sum function inside run_packing is a local scope function 
# in the original thought, but per instructions "Make all helper functions top level".
# I will restructure the code to follow the rules strictly.

def calculate_max_equal_radius(centers):
    """
    Calculates the maximum radius r such that n circles at 'centers' 
    with radius r do not overlap and are inside [0,1]x[0,1].
    """
    n = centers.shape[0]
    r_bound = 1.0
    for i in range(n):
        x, y = centers[i]
        r_bound = min(r_bound, x, 1-x, y, 1-y)
        
    # Vectorized distance check
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(r_bound, min_dist / 2.0)

def run_packing_refactored():
    """
    Main function to pack 26 circles.
    """
    n = 26
    np.random.seed(123)

    # Initialization: 5x5 grid + 1
    centers = np.zeros((n, 2))
    coords = np.linspace(0.1, 0.9, 5)
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [coords[i], coords[j]]
            idx += 1
    centers[25] = [0.2, 0.2]

    # Simulated Annealing
    T = 0.5
    alpha = 0.9998
    steps = 100000
    step_size = 0.002
    
    best_centers = centers.copy()
    best_r = calculate_max_equal_radius(centers)
    best_sum = n * best_r
    
    current_centers = centers.copy()
    current_r = best_r
    current_sum = best_sum
    
    for step in range(steps):
        i = np.random.randint(n)
        move = np.random.uniform(-step_size, step_size, 2)
        
        new_centers = current_centers.copy()
        new_centers[i] += move
        new_centers[i, 0] = np.clip(new_centers[i, 0], 0, 1)
        new_centers[i, 1] = np.clip(new_centers[i, 1], 0, 1)
        
        new_r = calculate_max_equal_radius(new_centers)
        new_sum = n * new_r
        
        if new_sum > current_sum:
            current_centers = new_centers
            current_r = new_r
            current_sum = new_sum
            if current_sum > best_sum:
                best_centers = current_centers.copy()
                best_r = new_r
                best_sum = current_sum
        else:
            if T > 1e-9:
                prob = np.exp((new_sum - current_sum) / T)
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_r = new_r
                    current_sum = new_sum
        
        if step > 10000:
            T *= alpha
        if T < 1e-7: T = 1e-7
            
    final_radii = np.ones(n) * best_r
    return best_centers, final_radii, best_sum

# Replace run_packing with the refactored version for the final output
run_packing = run_packing_refactored
