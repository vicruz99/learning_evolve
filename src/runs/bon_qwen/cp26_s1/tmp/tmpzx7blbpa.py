import numpy as np
import scipy.optimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Uses a hexagonal lattice initialization and nonlinear optimization.
    """
    n_circles = 26
    
    # Function to generate a hexagonal lattice configuration
    def generate_hex_lattice(n, width=1.0, height=1.0, padding=0.0):
        # We want to fit n points. 
        # A hexagonal lattice has spacing s.
        # Approximate number of points in area A with spacing s is A / (s^2 * sqrt(3)/2)
        # s^2 approx 2*A / (n*sqrt(3))
        # Area = (1-2*padding)^2
        avail_area = (width - 2*padding)**2
        # Estimate spacing s
        # Density of hex lattice is 2/(sqrt(3)*s^2) points per unit area? 
        # Area per point = s^2 * sqrt(3)/2
        # n * s^2 * sqrt(3)/2 approx avail_area
        # s = sqrt(2 * avail_area / (n * sqrt(3)))
        s = math.sqrt(2 * avail_area / (n * math.sqrt(3)))
        
        points = []
        # Grid generation
        # Rows spaced by s * sqrt(3)/2
        # Cols spaced by s
        # Alternating shift of s/2
        
        dy = s * math.sqrt(3) / 2
        y = padding + s # Start a bit inside? Or just padding?
        # Let's start at padding + r? No, coordinates are centers.
        # Let's just generate points in [0, 1] roughly and filter.
        
        # Better: Generate a large lattice and pick those inside [padding, 1-padding]
        # But we need exactly n.
        # Let's just create a grid and take the first n.
        
        # Let's construct rows
        row_idx = 0
        while len(points) < n:
            row_shift = (row_idx % 2) * (s / 2)
            y_coord = padding + s + row_idx * dy # Start y at padding + s to leave room?
            # Actually, let's just iterate y from padding
            # But hex packing needs specific y start.
            # Let's try: y starts at padding + s (radius margin approx s/2?)
            # Actually, optimal y for first row center is roughly r = s/2.
            # So y = padding + s/2?
            # Let's just generate a grid of potential centers.
            
            # Reset strategy:
            pass

    # Alternative: Just place points on a grid and let optimizer fix it?
    # No, hex is better.
    
    # Let's implement a robust hex generator
    def get_hex_config(n, padding=0.0):
        # Estimate spacing
        # Packing density ~ 0.9069
        # Area n * pi * r^2 <= 1 * 0.9069 => r <= 0.105
        # s = 2r <= 0.21
        # Let's start with s = 0.2
        s = 0.2 
        
        # Generate points
        pts = []
        # Rows
        y = s # center y for first row (approx r)
        row = 0
        while len(pts) < n:
            # Shift for hex
            shift = (s / 2.0) if (row % 2 == 1) else 0.0
            x = shift + s/2 # First point? 
            # Wait, boundaries.
            # Let's place points starting from s/2 (which is r)
            x = shift + s/2
            
            while True:
                if x + s/2 > 1.0 - padding:
                    break
                if x - s/2 < 0.0 + padding:
                    x += s # Move to next
                    continue
                
                pts.append((x, y))
                if len(pts) >= n:
                    break
                x += s
            
            if len(pts) >= n:
                break
            
            y += s * math.sqrt(3) / 2.0
            row += 1
            
            # Check height bound
            if y + s/2 > 1.0 - padding:
                break
        
        return np.array(pts[:n]), s

    # Try multiple strategies
    best_sum_r = -1.0
    best_centers = None
    best_radii = None

    # Strategy 1: Hexagonal Lattice Optimization
    # We will optimize r and positions.
    # Variables: [x1, y1, ..., x26, y26, r]
    # Total 53 variables.
    
    def objective(vars):
        # vars[0:52] are coords, vars[52] is r
        return -vars[52] # Minimize -r

    def boundary_constraints(vars):
        # x - r >= 0, 1 - (x + r) >= 0
        # y - r >= 0, 1 - (y + r) >= 0
        r = vars[52]
        coords = vars[:52].reshape(26, 2)
        con = []
        for i in range(26):
            con.append(coords[i, 0] - r) # x >= r
            con.append(coords[i, 1] - r) # y >= r
            con.append(1.0 - coords[i, 0] - r) # x <= 1-r
            con.append(1.0 - coords[i, 1] - r) # y <= 1-r
        return con

    def separation_constraints(vars):
        r = vars[52]
        coords = vars[:52].reshape(26, 2)
        con = []
        min_dist_sq = (2*r)**2
        for i in range(26):
            for j in range(i+1, 26):
                d_sq = np.sum((coords[i] - coords[j])**2)
                con.append(d_sq - min_dist_sq)
        return con

    def run_optimizer(initial_coords, init_r):
        initial_vars = np.concatenate([initial_coords.flatten(), [init_r]])
        
        bounds = []
        for i in range(52):
            bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5)) # r in [0, 0.5]
        
        # Constraints
        cons = []
        
        # We can't easily pass all separation constraints as a list of dicts for SLSQP in this loop 
        # without creating a huge list. 
        # SLSQP handles dict with 'fun' and 'jac' or list of dicts.
        # Creating 325 constraints might be slow but acceptable for N=26?
        # Let's try to define them.
        
        # To speed up, we can use a penalty method or just fewer constraints?
        # But for correctness, we need them.
        
        # Actually, defining 325 constraint functions is heavy.
        # Let's try a different approach: 
        # Fix r, optimize positions to maximize min distance.
        # Then binary search r.
        # Or use the repulsion simulation which is simpler to code and robust.
        
        # Repulsion Simulation Approach
        return None

    # Let's switch to Repulsion Simulation + Local Optimization
    # It's easier to implement without heavy constraint lists.
    
    def solve_with_repulsion(init_coords, init_r, iterations=2000, learning_rate=0.01):
        centers = init_coords.copy()
        r = init_r
        radii = np.full(26, r)
        
        # We want to maximize r.
        # Loop: increase r, resolve conflicts.
        
        step = 0.0001
        max_r = 0.0
        
        # Precompute pair indices
        pairs = []
        for i in range(26):
            for j in range(i+1, 26):
                pairs.append((i, j))
        
        # Initial check and expansion
        # Start with a safe r
        current_r = 0.01
        centers = init_coords * 0.5 + 0.5 # Center them? No, keep init.
        # Scale init_coords to fit small r
        # Ensure centers are within [r, 1-r]
        # Just normalize init_coords to [0.1, 0.9]
        centers = init_coords * 0.8 + 0.1
        
        # Run expansion
        # We try to increase r and push circles apart
        
        # Force constants
        k_repulse = 10.0
        k_wall = 5.0
        k_center = 0.0 # Maybe keep them from drifting? No, let them find spots.
        
        # Cooling schedule
        lr = 0.01
        
        for it in range(iterations):
            # Increase r slightly
            current_r += step / (1 + it/100) # Slow down increase
            if current_r > 0.2: current_r = 0.2 # Cap roughly
            
            radii[:] = current_r
            
            # Compute forces
            forces = np.zeros((26, 2))
            
            # Pairwise repulsion
            for i, j in pairs:
                d_vec = centers[i] - centers[j]
                dist = np.linalg.norm(d_vec)
                if dist < 1e-9:
                    dist = 1e-9
                    d_vec = np.random.rand(2) * 0.01 # Random jitter
                
                min_dist = 2.0 * current_r
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    # F = (min_dist - dist) / dist * direction
                    # Simple spring force
                    force_mag = k_repulse * (min_dist - dist)
                    force_vec = force_mag * (d_vec / dist)
                    forces[i] += force_vec
                    forces[j] -= force_vec
            
            # Wall repulsion
            for i in range(26):
                x, y = centers[i]
                r = current_r
                
                # Left wall
                if x - r < 0:
                    forces[i, 0] += k_wall * (r - x)
                # Right wall
                elif x + r > 1:
                    forces[i, 0] -= k_wall * (x + r - 1)
                
                # Bottom wall
                if y - r < 0:
                    forces[i, 1] += k_wall * (r - y)
                # Top wall
                elif y + r > 1:
                    forces[i, 1] -= k_wall * (y + r - 1)
            
            # Apply forces
            centers += forces * lr
            
            # Clamp centers to valid range (hard constraint)
            # Actually, let forces handle it, but clamp to [r, 1-r] roughly?
            # Clamping might destroy momentum.
            # Just clamp to [0, 1] to keep them in square.
            centers = np.clip(centers, 0.0, 1.0)
            
            # Check if stable?
            # If max force is small, we might be stuck or stable.
            # But we are increasing r, so we are always pushing.
            
            # Adaptive learning rate?
            if it > 1000:
                lr *= 0.995
            
            # Check validity roughly
            # If any overlap is huge, reduce r? 
            # No, we want to find max r.
            
            # Record best valid r?
            # The simulation might overshoot.
            # We should check validity periodically.
            if it % 100 == 0:
                valid = True
                min_dist_found = 1.0
                for i, j in pairs:
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 2*current_r - 1e-7:
                        valid = False
                        break
                    min_dist_found = min(min_dist_found, d)
                
                for i in range(26):
                    x, y = centers[i]
                    if x < current_r - 1e-7 or x > 1 - current_r + 1e-7 or \
                       y < current_r - 1e-7 or y > 1 - current_r + 1e-7:
                        valid = False
                        break
                
                if valid:
                    max_r = current_r
                    # Save state?
        
        # After simulation, the current_r might be too high (violating constraints)
        # We need to find the largest r that is valid for the final positions.
        # But positions might be suboptimal for that r.
        # Actually, the simulation tries to satisfy constraints for current_r.
        # If current_r is increased too fast, constraints are violated.
        # We should decrease r until valid, then check if we can go higher?
        
        # Better: The simulation state at the end is a local optimum for some r.
        # Let's compute the actual feasible r for the final configuration.
        
        feasible_r = 1.0
        # Distance to boundaries
        for i in range(26):
            x, y = centers[i]
            feasible_r = min(feasible_r, x, 1-x, y, 1-y)
        
        # Distance between circles
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            feasible_r = min(feasible_r, d / 2.0)
        
        # Return this configuration
        return centers, np.full(26, feasible_r), feasible_r * 26

    # Helper to generate hex lattice
    def create_hex_lattice(n, scale=1.0):
        # Generate points in a large grid
        # We want roughly n points
        # Estimate s
        # Area ~ 1. s ~ 1/sqrt(n)
        s = 1.0 / math.sqrt(n) * 0.9 # Slightly tighter
        
        pts = []
        y = s # Start y
        row = 0
        while len(pts) < n:
            shift = (s/2.0) if (row % 2 == 1) else 0.0
            x = shift + s/2.0 # Center of first cell?
            # Let's just iterate
            while True:
                if x + s/2 > 1.0: break # Out of bounds x
                if x - s/2 < 0.0: # Out of bounds x
                    x += s
                    continue
                
                # Check y
                if y + s/2 > 1.0 or y - s/2 < 0.0:
                    break # Out of bounds y (for this row)
                
                pts.append((x, y))
                x += s
            
            y += s * math.sqrt(3) / 2.0
            row += 1
        
        return np.array(pts[:n])

    # Run simulation with different initial configs
    configs = []
    
    # 1. Hex lattice
    hex_pts = create_hex_lattice(26)
    configs.append(hex_pts)
    
    # 2. Perturbed Hex
    configs.append(hex_pts + np.random.randn(26, 2) * 0.05)
    
    # 3. Grid 6x5 (subset)
    grid_pts = []
    # 6 cols, 5 rows = 30 points. Take 26.
    xs = np.linspace(0.15, 0.85, 6) # 6 points
    ys = np.linspace(0.15, 0.85, 5) # 5 points
    # Actually linspace 0.1 to 0.9
    xs = np.linspace(0.1, 0.9, 6) 
    ys = np.linspace(0.1, 0.9, 5)
    for y in ys:
        for x in xs:
            grid_pts.append([x, y])
            if len(grid_pts) >= 26:
                break
        if len(grid_pts) >= 26:
            break
    configs.append(np.array(grid_pts))
    
    # 4. Random
    configs.append(np.random.rand(26, 2))
    
    # 5. Random in circle
    angles = np.random.rand(26) * 2 * np.pi
    radii_rand = np.sqrt(np.random.rand(26)) * 0.45
    cx, cy = 0.5, 0.5
    rand_pts = np.column_stack([cx + radii_rand * np.cos(angles), cy + radii_rand * np.sin(angles)])
    configs.append(rand_pts)

    best_solution = None
    best_sum = -1.0
    
    for i, init_c in enumerate(configs):
        # Ensure init_c is valid-ish (inside square)
        init_c = np.clip(init_c, 0.01, 0.99)
        
        # Run repulsion
        # Use fixed seed for reproducibility if needed, but random is fine.
        # We run multiple times?
        # Let's run once per config.
        
        # Optimization parameters tuning
        # Try to run for longer
        res_centers, res_radii, res_sum = solve_with_repulsion(init_c, 0.05, iterations=3000)
        
        # Verify validity
        # The solve_with_repulsion computes feasible_r based on distances.
        # So it should be valid.
        # But let's double check.
        
        # Check overlap
        valid = True
        for a in range(26):
            for b in range(a+1, 26):
                d = np.sqrt(np.sum((res_centers[a] - res_centers[b])**2))
                if d < res_radii[a] + res_radii[b] - 1e-12:
                    valid = False
                    break
            if not valid: break
        
        # Check boundaries
        if valid:
            for a in range(26):
                x, y = res_centers[a]
                r = res_radii[a]
                if x < r - 1e-12 or x > 1-r + 1e-12 or y < r - 1e-12 or y > 1-r + 1e-12:
                    valid = False
                    break
        
        if valid:
            if res_sum > best_sum:
                best_sum = res_sum
                best_solution = (res_centers, res_radii)
        
        # Try to improve with local search (gradient based) if valid?
        # Maybe not needed if repulsion is good.
        
        # Run again with different parameters?
        # Repulsion can get stuck.
        # Let's try a second pass with smaller step?
        
    if best_solution is None:
        # Fallback to grid
        centers = np.array(grid_pts)
        radii = np.full(26, 0.05) # Conservative
        # Adjust radii to be valid
        # ...
        # Just return something valid
        best_solution = (np.array(grid_pts), np.full(26, 0.01))
        best_sum = 0.26

    centers, radii = best_solution
    
    # Final validation check and potential tweak
    # The radii are all equal in our solver.
    # We can try to increase individual radii in gaps?
    # But sum of radii with equal radii is likely near optimal.
    # However, we can check if we can increase any radius.
    
    # Check if we can increase radius of circle i
    # This is a complex optimization (max r_i s.t. constraints).
    # Given time constraints, we stick to the equal radius result.
    # But wait, the solver finds a configuration for a specific r.
    # It computes feasible_r which is the bottleneck.
    # All circles have radius feasible_r.
    # Some circles might be loose.
    # But sum is 26 * feasible_r.
    
    # Is it possible that unequal radii yield higher sum?
    # Maybe. But hard to solve.
    # Let's hope equal is good enough.
    
    # Refine the solution:
    # We can try to push circles apart using scipy.optimize on the current valid config.
    # We want to minimize a penalty function.
    # Penalty = sum(max(0, r_i + r_j - dist_ij)^2) + sum(boundary penalties)
    # And we want to maximize sum(r_i).
    # This is equivalent to: Maximize sum(r_i) - lambda * Penalty.
    # Or just use the constraints.
    
    # Let's try a quick local optimization on radii?
    # If we have extra space, we can increase radii.
    # But our solver already maximized r for the configuration.
    # The configuration was optimized for that r.
    # So it's likely optimal for that shape.
    
    # One trick: The repulsion simulation might not be perfectly converged.
    # We can run a few more steps of "pushing" if valid.
    # But the function solve_with_repulsion runs for 3000 steps.
    
    # Let's ensure the returned values are numpy arrays.
    centers = np.array(centers)
    radii = np.array(radii)
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r

# To be safe, execute the function logic inside run_packing
# But we need to return the tuple.
# The code above defines run_packing.

# One issue: create_hex_lattice might return < 26 points if bounds are tight?
# My implementation checks bounds.
# If it returns < 26, we have a problem.
# Let's fix create_hex_lattice to ensure 26 points.
# It should generate enough points.
# With s ~ 0.2, it should fit many.
# If it fails, we fallback to grid.

# Let's rewrite run_packing cleanly.

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    def get_hex_pts(n):
        pts = []
        # Spacing estimate
        s = 0.2 
        # We need to fit n points.
        # Let's generate a grid of potential centers and pick valid ones?
        # Or just place them.
        
        # Try to fit in [0.1, 0.9] roughly
        # Rows
        y = 0.1
        row = 0
        while len(pts) < n:
            shift = 0.0
            if row % 2 == 1:
                shift = s / 2.0
            x = 0.1 + shift
            while x <= 0.9 and len(pts) < n:
                pts.append([x, y])
                x += s
            y += s * math.sqrt(3) / 2.0
            if y > 0.9 + s/2: # Check if we are going too high
                # If we haven't found n points, reduce s or relax bounds
                pass
            row += 1
        
        # If we didn't get n points, fill with random?
        while len(pts) < n:
            pts.append([np.random.rand(), np.random.rand()])
            
        return np.array(pts[:n])

    # 2. Repulsion Simulation
    def optimize_packing(centers_init):
        centers = centers_init.copy()
        r = 0.01 # Start small
        radii = np.full(n, r)
        
        # Precompute pairs
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Parameters
        k_rep = 5.0
        k_wall = 10.0
        lr = 0.005
        
        # We want to find the largest r.
        # Strategy: Slowly increase r, apply forces to resolve conflicts.
        # If conflict cannot be resolved (overlap > threshold), reduce r?
        # No, just push hard.
        
        # Initial placement: ensure inside [r, 1-r]
        # Clip to [0.05, 0.95]
        centers = np.clip(centers, 0.05, 0.95)
        
        # Current target r
        target_r = 0.05
        
        for step in range(5000):
            # Increase target r slowly
            # Decay learning rate
            if step > 2000:
                lr *= 0.999
            if step % 100 == 0:
                target_r += 0.0005 # Increase radius requirement
                radii[:] = target_r
            
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            # Vectorized loop for speed?
            # n=26 is small, python loop is fine.
            for i, j in pairs:
                c_i = centers[i]
                c_j = centers[j]
                diff = c_i - c_j
                dist_sq = np.dot(diff, diff)
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-9
                
                min_dist = 2.0 * target_r
                if dist < min_dist:
                    # Repulsion
                    # Force magnitude proportional to overlap
                    overlap = min_dist - dist
                    # Normalize diff
                    if dist > 1e-9:
                        unit_diff = diff / dist
                        force_mag = k_rep * overlap
                        f = force_mag * unit_diff
                        forces[i] += f
                        forces[j] -= f
            
            # Wall repulsion
            for i in range(n):
                x, y = centers[i]
                r = target_r
                
                # X walls
                if x < r:
                    forces[i, 0] += k_wall * (r - x)
                elif x > 1.0 - r:
                    forces[i, 0] -= k_wall * (x - (1.0 - r))
                
                # Y walls
                if y < r:
                    forces[i, 1] += k_wall * (r - y)
                elif y > 1.0 - r:
                    forces[i, 1] -= k_wall * (y - (1.0 - r))
            
            # Update centers
            centers += forces * lr
            
            # Clamp to [0, 1] strictly
            centers = np.clip(centers, 0.0, 1.0)
            
            # Check convergence? 
            # We just run fixed steps.
            
        # After simulation, compute valid radii for each circle
        # We found a configuration. What is the max radius for EACH circle?
        # Actually, we enforced equal radii target_r during sim.
        # But the final state might support unequal radii or a different equal radius.
        # Let's compute the bottleneck radius for equal radii.
        
        max_equal_r = 1.0
        
        # Boundary limits
        for i in range(n):
            x, y = centers[i]
            max_equal_r = min(max_equal_r, x, 1-x, y, 1-y)
        
        # Pairwise limits
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            max_equal_r = min(max_equal_r, d / 2.0)
        
        # Set all radii to this max_equal_r
        radii = np.full(n, max_equal_r)
        
        return centers, radii

    # Try multiple starts
    best_sum = -1
    best_c = None
    best_r = None
    
    starts = [
        get_hex_pts(n),
        np.random.rand(n, 2) * 0.8 + 0.1, # Random centered
    ]
    
    # Add a grid start
    grid_x = np.linspace(0.1, 0.9, 6)
    grid_y = np.linspace(0.1, 0.9, 5)
    grid_pts = []
    for y in grid_y:
        for x in grid_x:
            grid_pts.append([x, y])
            if len(grid_pts) >= n: break
        if len(grid_pts) >= n: break
    starts.append(np.array(grid_pts))
    
    for start in starts:
        # Perturb slightly
        start_perturbed = start + np.random.randn(*start.shape) * 0.02
        start_perturbed = np.clip(start_perturbed, 0.05, 0.95)
        
        c, r = optimize_packing(start_perturbed)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_c = c
            best_r = r
            
    return best_c, best_r, best_sum