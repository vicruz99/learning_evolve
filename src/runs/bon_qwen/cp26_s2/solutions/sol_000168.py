# sol_000168 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d2f1ce33) state=daae995d sum of radii=0.016398 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Number of circles
    n = 26
    
    # Initial guess: 5x5 grid + 1 circle in center
    # We'll use a slightly perturbed grid to allow optimization to move things
    centers = np.zeros((n, 2))
    
    # 5x5 grid centers
    coords = np.linspace(0.1, 0.9, 5)
    count = 0
    for i in range(5):
        for j in range(5):
            if count < n:
                centers[count] = [coords[i], coords[j]]
                count += 1
    
    # If we need more circles (26), place the last one near center or in a gap
    # 25 circles filled. Place 26th at (0.5, 0.5) which is occupied, 
    # so let's perturb it slightly or place it in a corner gap.
    # Actually, (0.5, 0.5) is a center in 5x5 grid?
    # linspace(0.1, 0.9, 5) -> [0.1, 0.3, 0.5, 0.7, 0.9]. Yes, 0.5 is there.
    # So we have 25 circles. We need 26.
    # Let's remove the center one (0.5, 0.5) and put 2 circles there?
    # Or just perturb the whole set.
    # Better initialization: 5 rows. 
    # Maybe 6-5-6-5-4 pattern? 
    # Let's stick to a dense random perturbation of a grid.
    
    # Let's try a configuration that is known to be dense.
    # A rotated grid might be better.
    # Let's initialize with a grid, then rotate slightly?
    # Or just let the optimizer figure it out from a random dense config.
    
    # Let's create a 5x5 grid but shift rows slightly to make room?
    # No, let's just use the grid and add a tiny circle, then optimize equal radii.
    # But 26th circle must be placed.
    # Let's replace the center circle (0.5, 0.5) with two circles? No, N is fixed.
    # Let's just place the 26th circle at (0.5, 0.5) overlapping, 
    # and let the optimizer resolve it by reducing r and moving.
    # But (0.5, 0.5) is occupied. 
    # Let's place it at (0.5, 0.52) or something.
    
    # Better: Initialize with a hexagonal packing pattern if possible, 
    # or a perturbed grid.
    # Let's generate a grid and add noise.
    np.random.seed(42)
    
    # Grid generation
    x_coords = np.linspace(0.1, 0.9, 5)
    y_coords = np.linspace(0.1, 0.9, 5)
    grid_centers = np.array([[x, y] for x in x_coords for y in y_coords]) # 25 points
    
    # We need 26. Let's add one near the center but slightly offset?
    # Or distribute 26 points more evenly.
    # Let's try to fit 26 points in a square.
    # Maybe 6 in first row, 5 in second...
    # Row 1 (y=0.1): 6 points? x from 0.08 to 0.92?
    # Let's try a specific layout: 
    # 6 circles in row 1, 5 in row 2, 6 in row 3, 5 in row 4, 4 in row 5?
    # Total 26.
    # This is a common pattern for packing.
    
    # Let's build this layout manually for initialization
    init_centers = []
    
    # Row parameters
    # We want to fit width 1.
    # If 6 circles, spacing ~ 1/6?
    # Let's just place them roughly.
    
    # Row 0: 6 circles
    y = 0.1
    xs = np.linspace(1/12, 11/12, 6) # roughly
    for x in xs: init_centers.append([x, y])
    
    # Row 1: 5 circles, shifted
    y = 0.1 + 0.1 * math.sqrt(3) # ~0.273
    xs = np.linspace(1/12 + 0.083, 11/12 - 0.083, 5) # shift
    # Actually simpler: centers at 2/12, 4/12...
    xs = [2/12, 4/12, 6/12, 8/12, 10/12]
    for x in xs: init_centers.append([x, y])
    
    # Row 2: 6 circles
    y = 0.1 + 2 * 0.1 * math.sqrt(3) # ~0.446
    xs = np.linspace(1/12, 11/12, 6)
    for x in xs: init_centers.append([x, y])
    
    # Row 3: 5 circles
    y = 0.1 + 3 * 0.1 * math.sqrt(3) # ~0.619
    xs = [2/12, 4/12, 6/12, 8/12, 10/12]
    for x in xs: init_centers.append([x, y])
    
    # Row 4: 4 circles
    y = 0.1 + 4 * 0.1 * math.sqrt(3) # ~0.792
    xs = [3/12, 5/12, 7/12, 9/12]
    for x in xs: init_centers.append([x, y])
    
    # Check count
    # 6 + 5 + 6 + 5 + 4 = 26.
    
    # This layout is quite wide (6 circles). 
    # Width needed for 6 circles touching: 12r.
    # If r=0.1, width=1.2 > 1.
    # So this initialization is invalid for r=0.1.
    # But optimizer will shrink r.
    # However, we want r ~ 0.101.
    # A 6-circle row is bad for r=0.1.
    
    # Let's go back to 5x5 grid + 1.
    # But where to put the 26th?
    # If we perturb the 5x5 grid, maybe we can squeeze one in.
    # Let's initialize with 26 circles in a perturbed 5x5 grid.
    # 5x5 has 25. 
    # Maybe split the center circle?
    # No, just random perturbation of a dense grid.
    
    # Let's try to generate 26 points using a Poisson disk sampling or just grid with noise.
    # Or simply: 5 rows of 5, plus 1.
    # Place 25 in 5x5 grid with r=0.09 (to leave room).
    # Place 26th at (0.5, 0.5).
    
    centers = np.zeros((n, 2))
    r_init = 0.09
    
    # 5x5 grid
    pts = np.array([[i*0.2 + 0.1, j*0.2 + 0.1] for i in range(5) for j in range(5)])
    centers[:25] = pts
    # 26th at center
    centers[25] = [0.5, 0.5]
    
    # Now optimize
    # We optimize r and centers.
    # Variables: [x0, y0, ..., x25, y25, r]
    # Total 53 variables.
    
    def objective(vars):
        # vars[:52] are centers, vars[52] is r
        return -vars[52] # Minimize -r => Maximize r
    
    def constraint_boundary(vars):
        # r <= x <= 1-r  =>  x - r >= 0  and  1 - r - x >= 0
        # y constraints similar
        # Inequality constraints g(x) >= 0 for SLSQP? 
        # SLSQP uses 'ineq' where g(x) >= 0.
        r = vars[52]
        cs = vars[:52].reshape(26, 2)
        constraints = []
        for i in range(26):
            x, y = cs[i]
            constraints.append(x - r)      # x >= r
            constraints.append(1.0 - r - x) # 1-r >= x => x <= 1-r
            constraints.append(y - r)
            constraints.append(1.0 - r - y)
        return np.array(constraints)
    
    def constraint_overlap(vars):
        r = vars[52]
        cs = vars[:52].reshape(26, 2)
        constraints = []
        for i in range(26):
            for j in range(i + 1, 26):
                dist = np.sqrt(np.sum((cs[i] - cs[j])**2))
                # dist >= 2r => dist - 2r >= 0
                constraints.append(dist - 2*r)
        return np.array(constraints)
    
    # Initial variables
    x0 = np.hstack((centers.flatten(), [r_init]))
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(26):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    bounds.append((0.0, 0.5)) # r
    
    # Constraints dict
    cons = []
    cons.append({'type': 'ineq', 'fun': constraint_boundary})
    # Overlap constraint is computationally expensive (325 constraints).
    # For SLSQP, passing a large array is okay, but calculating Jacobian is hard.
    # SLSQP approximates Jacobian.
    # However, 325 constraints might be slow.
    # Let's try it. If it fails, we might need a simpler approach.
    # But 26 circles is small enough.
    
    # Actually, SLSQP might struggle with 325 inequality constraints and non-smooth objective (min of distances?).
    # dist is smooth except at 0. But circles won't be at same location.
    
    # Let's define overlap constraint.
    # To speed up, we can use a penalty function in objective?
    # But SLSQP handles constraints.
    
    # Optimization
    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, 
                           options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
        
        if res.success:
            best_r = res.x[52]
            best_centers = res.x[:52].reshape(26, 2)
        else:
            # If optimization fails, return initial or try again
            best_r = r_init
            best_centers = centers
            
    except Exception as e:
        print(f"Optimization error: {e}")
        best_r = r_init
        best_centers = centers

    # Post-processing: Ensure valid packing
    # The optimizer might return slightly invalid due to tolerance.
    # We can shrink radii slightly to ensure validity.
    # Or just return as is if valid.
    
    # Let's verify and adjust if needed
    radii = np.full(26, best_r)
    
    # Check validity
    # If invalid, we can't easily fix it without re-optimizing.
    # But we can just return the result. The validation function has tolerance 1e-12.
    
    # Let's try to improve with a local search if needed?
    # The SLSQP should be sufficient.
    
    # Just in case SLSQP is slow or fails, let's implement a simple gradient descent / simulated annealing
    # fallback or just run SLSQP.
    # Given the complexity, SLSQP is the best bet in scipy.
    
    # To be safe, let's run a quick check and if sum is low, maybe try a different init?
    # But we only have one shot in run_packing.
    
    # Let's refine the solution.
    # If best_r is around 0.1, sum is 2.6.
    # We need 2.636.
    # SLSQP might get stuck.
    # Let's try to run it multiple times with different seeds?
    # No, time limit.
    
    # Let's use a simpler heuristic:
    # Start with 26 circles in a grid (perturbed).
    # Use a force-directed relaxation (repulsion) to push them apart, 
    # then scale up.
    # This is often more robust for packing.
    
    # Force-directed approach:
    # 1. Initialize centers.
    # 2. Iteratively move centers to maximize min distance.
    # 3. Compute r = min_dist / 2.
    # 4. Check boundaries.
    
    # This is equivalent to finding the "max-min distance" problem (Tammes problem variant).
    # Once we have optimal positions for min distance, r is determined.
    # Then sum = 26 * r.
    
    # Let's implement a simple force-directed optimizer.
    
    # Re-initialize
    np.random.seed(123)
    # Initialize with a grid perturbed
    centers = np.random.uniform(0.05, 0.95, (26, 2))
    # Or better: 5x5 grid + 1
    coords = np.linspace(0.1, 0.9, 5)
    pts = np.array([[x, y] for x in coords for y in coords])
    centers = pts.copy()
    centers = np.vstack([centers, [0.5, 0.5]]) # 26th at center
    
    # Force simulation
    # Repulsive force between circles: F = 1 / d^2 (or similar)
    # Constraint forces for boundaries.
    
    # We want to maximize the minimum distance between any pair (and to walls).
    # Let's use a simple iterative repulsion.
    
    learning_rate = 0.01
    
    for step in range(500): # 500 iterations
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(26):
            for j in range(i + 1, 26):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    dist = 1e-6
                    diff = np.random.rand(2) * 0.01 # Kick
                
                # We want dist to be large.
                # Force proportional to 1/dist or similar.
                # To maximize min dist, we can push apart if dist is small.
                # But actually, we want to push ALL apart.
                # F = k / dist^2 * direction
                force_mag = 1.0 / (dist**2) 
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Boundary forces (keep inside [0,1])
        # Soft constraint: if close to boundary, push in.
        # Actually, hard constraint: clip?
        # Or force.
        # Let's use soft force from boundaries.
        # Boundary at 0: if x < eps, push right.
        # But we want to fill the square.
        # The repulsion will naturally push them to corners.
        # Just ensure they stay inside.
        
        centers += learning_rate * forces
        centers = np.clip(centers, 0.0, 1.0) # Hard clamp
        
        # Reduce learning rate
        learning_rate *= 0.99
        
    # After simulation, find the optimal radius
    # r is limited by:
    # 1. Distance to walls: min(x, 1-x, y, 1-y)
    # 2. Distance to neighbors: min(dist(i,j))/2
    
    min_dist_wall = min(np.min(centers[:, 0]), 
                        np.min(1 - centers[:, 0]), 
                        np.min(centers[:, 1]), 
                        np.min(1 - centers[:, 1]))
    
    min_dist_pair = np.inf
    for i in range(26):
        for j in range(i + 1, 26):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist_pair:
                min_dist_pair = d
    
    r_final = min(min_dist_wall, min_dist_pair / 2.0)
    
    # This r_final might be small if simulation didn't converge well.
    # But force-directed usually finds good packings.
    
    # Let's try to refine this r_final.
    # The centers might not be optimal for the *found* r_final.
    # But it's a good approximation.
    
    # To improve, we can run a few steps of "expand" simulation.
    # But the static calculation above gives a valid packing.
    
    # Let's verify validity
    radii = np.full(26, r_final)
    
    # Check overlaps
    valid = True
    for i in range(26):
        for j in range(i + 1, 26):
            if np.linalg.norm(centers[i] - centers[j]) < 2 * r_final - 1e-12:
                valid = False
    
    # Check boundaries
    for i in range(26):
        x, y = centers[i]
        if x < r_final - 1e-12 or x > 1 - r_final + 1e-12: valid = False
        if y < r_final - 1e-12 or y > 1 - r_final + 1e-12: valid = False
        
    if not valid:
        # Shrink slightly
        r_final *= 0.999
        
    sum_r = np.sum(radii)
    
    # If sum_r is low, maybe try a different init?
    # But we can't retry easily.
    # Let's hope the force simulation works.
    # 500 steps might be few.
    
    # Let's increase steps or use a better init.
    # Better init: 5x5 grid is good.
    # But 26th circle at center causes overlap.
    # Force simulation will push them apart.
    
    # Let's combine: 
    # 1. Force simulation to spread points.
    # 2. Calculate max r.
    
    # To ensure high quality, maybe run multiple restarts?
    # No time.
    
    # Let's try to run the SLSQP on the result of force simulation?
    # Force simulation gives good centers. 
    # Then we can optimize r?
    # Actually, if we have centers, r is fixed by min distance.
    # We can move centers to increase min distance.
    # This is exactly what SLSQP did (maximize r).
    # But SLSQP with 325 constraints is hard.
    # Maybe we can optimize r locally?
    
    # Let's stick to the force simulation result, but maybe run it longer.
    # Or use a library function if available? No.
    
    # Let's refine the force simulation code.
    # Using a repulsive force 1/d is standard for "spread out points".
    
    # Let's re-run the logic with a better loop.
    
    # Re-init with grid
    np.random.seed(10)
    centers = np.random.uniform(0.0, 1.0, (26, 2)) # Random start is also good?
    # Grid start is better for structure.
    # Let's stick to grid + noise.
    grid_pts = np.array([[i*0.2 + 0.1, j*0.2 + 0.1] for i in range(5) for j in range(5)])
    centers = grid_pts.copy()
    # Add 26th
    # Find the largest empty spot?
    # Center (0.5, 0.5) is occupied.
    # (0.5, 0.5) is at index 12 (3rd row, 3rd col).
    # Let's move the center one slightly?
    # Or just add at (0.5, 0.5) and let forces sort it.
    centers = np.vstack([centers, [0.5, 0.5]])
    
    # Simulation
    lr = 0.05
    for _ in range(2000):
        forces = np.zeros_like(centers)
        for i in range(26):
            for j in range(i+1, 26):
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-5:
                    d = 1e-5
                    d_vec = np.random.randn(2) * 0.01
                # Force magnitude. 
                # We want to maximize min distance. 
                # Force ~ 1/d is good.
                f = 1.0 / d
                forces[i] += (d_vec / d) * f
                forces[j] -= (d_vec / d) * f
        
        # Wall repulsion
        # Push away from 0 and 1
        # F_wall = 1 / dist_to_wall
        dist_left = centers[:, 0]
        dist_right = 1.0 - centers[:, 0]
        dist_bottom = centers[:, 1]
        dist_top = 1.0 - centers[:, 1]
        
        # Soft forces
        # If dist < 0.2, apply force
        mask_l = dist_left < 0.2
        forces[mask_l, 0] += 0.5 / (dist_left[mask_l] + 0.01)
        
        mask_r = dist_right < 0.2
        forces[mask_r, 0] -= 0.5 / (dist_right[mask_r] + 0.01)
        
        mask_b = dist_bottom < 0.2
        forces[mask_b, 1] += 0.5 / (dist_bottom[mask_b] + 0.01)
        
        mask_t = dist_top < 0.2
        forces[mask_t, 1] -= 0.5 / (dist_top[mask_t] + 0.01)
        
        centers += lr * forces
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        lr *= 0.995
        
    # Calculate r
    min_d = np.inf
    for i in range(26):
        for j in range(i+1, 26):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_d: min_d = d
    
    r = min_d / 2.0
    
    # Check walls
    wall_constr = np.min(np.concatenate([centers[:, 0], 1-centers[:, 0], centers[:, 1], 1-centers[:, 1]]))
    if wall_constr < r:
        r = wall_constr
        
    radii = np.full(26, r)
    
    # Final check and slight shrink if needed
    # Validate manually to be sure
    # (Skipping full validation loop for speed, trusting math)
    # But let's do a quick check
    valid = True
    for i in range(26):
        x,y = centers[i]
        if x < r - 1e-9 or x > 1-r + 1e-9 or y < r - 1e-9 or y > 1-r + 1e-9:
            valid = False
    for i in range(26):
        for j in range(i+1, 26):
            if np.linalg.norm(centers[i]-centers[j]) < 2*r - 1e-9:
                valid = False
    
    if not valid:
        r *= 0.9999
        radii = np.full(26, r)
        
    return centers, radii, np.sum(radii)
