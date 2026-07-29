# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=976ec6b5 sum of radii=2.300113 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed local optimization starting from a hexagonal grid.
    """
    n_circles = 26
    num_iterations = 2000
    num_steps_per_grow = 50
    
    # Initialize centers in a hexagonal pattern
    # We estimate an initial radius of 0.1 to generate a grid
    r_est = 0.1
    centers = []
    
    # Generate points on a hexagonal lattice
    # Spacing x: 2*r_est, Spacing y: sqrt(3)*r_est
    dy = math.sqrt(3) * r_est
    
    # We want to cover the square. 
    # Let's generate a grid slightly finer than needed and pick points,
    # or just construct a specific pattern.
    # A 5x5 grid is 25 points. We need 1 more.
    # Hexagonal packing allows more.
    # Let's generate points in rows.
    
    y = r_est
    row = 0
    while y + r_est <= 1.0 + 1e-9:
        # Determine x offset for this row (0 for even rows, r_est for odd)
        x_offset = r_est if row % 2 == 1 else r_est
        x = x_offset
        while x + r_est <= 1.0 + 1e-9:
            centers.append([x, y])
            x += 2 * r_est
        y += dy
        row += 1
    
    # If we have more than 26 points, we need to select 26.
    # If fewer, we need to add more.
    # With r_est=0.1, we likely get around 23-25 points in the interior.
    # Let's ensure we have at least 26 points.
    # If the grid generation is sparse, we can just add points randomly or fill gaps.
    
    # Better initialization: Randomly place 26 points in [r_est, 1-r_est]^2
    # But structured is better.
    # Let's try to force 26 points.
    # If the grid above gives < 26, we can lower r_est to generate more, then optimize.
    # But optimization will push them apart anyway.
    
    # Let's try a specific pattern for 26: 5 rows of 5, 5, 5, 5, 4, 2? 
    # Or just take the generated points. If < 26, add random points.
    
    current_centers = np.array(centers)
    if len(current_centers) < n_circles:
        # Add random points to fill up to 26
        np.random.seed(42) # For reproducibility
        while len(current_centers) < n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            current_centers = np.vstack([current_centers, [x, y]])
        current_centers = current_centers[:n_circles]
    elif len(current_centers) > n_circles:
        # Select 26 points that are most spread out? 
        # Just taking the first 26 might be biased towards corners.
        # Let's just take first 26 for now, optimization will fix it.
        current_centers = current_centers[:n_circles]

    # Force-directed optimization
    # We will try to maximize the radius r such that all circles fit.
    # We grow r and repel circles.
    
    r_current = 0.02
    r_target = 0.02
    
    # We will iterate, increasing r_target slowly and relaxing positions
    max_r_target = 0.12 # Upper bound estimate
    
    dt = 1e-4 # Time step for integration
    damping = 0.95 # Velocity damping
    
    velocities = np.zeros_like(current_centers)
    
    # To make optimization smoother, we can run multiple passes
    for pass_num in range(3):
        r_target = 0.02
        step_size = (max_r_target - r_target) / 500.0
        
        # If we already found a large radius in previous pass, start there
        if pass_num > 0:
            # Estimate current packing radius from min distance
            dists = np.array([np.linalg.norm(current_centers[i] - current_centers[j]) 
                              for i in range(n_circles) for j in range(i+1, n_circles)])
            min_dist = np.min(dists)
            # Also check boundaries
            dists_boundary = np.minimum(np.minimum(current_centers[:, 0], 1 - current_centers[:, 0]),
                                        np.minimum(current_centers[:, 1], 1 - current_centers[:, 1]))
            min_dist_b = np.min(dists_boundary)
            r_start = min(min_dist / 2.0, min_dist_b)
            r_target = max(0.02, r_start)
            
        # Optimization loop
        for _ in range(2000):
            # Increase target radius
            r_target += step_size * 0.5 # Slow growth
            
            # Compute forces
            forces = np.zeros_like(current_centers)
            
            # Circle-circle repulsion
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    diff = current_centers[i] - current_centers[j]
                    dist = np.linalg.norm(diff)
                    min_dist_req = 2 * r_target
                    
                    if dist < min_dist_req and dist > 1e-6:
                        # Repulsive force proportional to overlap
                        overlap = min_dist_req - dist
                        # Direction is normalized diff
                        direction = diff / dist
                        # Force magnitude: stiff spring
                        f_mag = overlap * 50.0 
                        forces[i] += direction * f_mag
                        forces[j] -= direction * f_mag
            
            # Boundary repulsion
            for i in range(n_circles):
                x, y = current_centers[i]
                # Left wall
                if x < r_target:
                    forces[i, 0] += (r_target - x) * 100.0
                # Right wall
                if x > 1 - r_target:
                    forces[i, 0] -= (x - (1 - r_target)) * 100.0
                # Bottom wall
                if y < r_target:
                    forces[i, 1] += (r_target - y) * 100.0
                # Top wall
                if y > 1 - r_target:
                    forces[i, 1] -= (y - (1 - r_target)) * 100.0
            
            # Update velocities and positions
            velocities += forces * dt
            velocities *= damping
            current_centers += velocities * dt
            
            # Clamp positions to square just in case
            current_centers[:, 0] = np.clip(current_centers[:, 0], 0, 1)
            current_centers[:, 1] = np.clip(current_centers[:, 1], 0, 1)

    # After optimization, compute exact maximum radii for the final positions
    # This is a linear programming problem, but for small N we can approximate or solve iteratively.
    # Or simply, since we optimized for a uniform radius r_target, 
    # the actual feasible radius might be slightly different.
    # However, the positions are optimized for r_target.
    # Let's calculate the max feasible radius for each circle given the fixed centers.
    # r_i <= dist(center_i, boundary)
    # r_i + r_j <= dist(center_i, center_j)
    
    # A simple greedy assignment or solving the system:
    # We can use an iterative method to find r_i
    r_radii = np.zeros(n_circles)
    
    # Initialize radii with boundary constraints
    for i in range(n_circles):
        x, y = current_centers[i]
        r_radii[i] = min(x, 1-x, y, 1-y)
    
    # Relax radii to satisfy pairwise constraints
    # This is finding the largest vector r such that r_i + r_j <= d_ij
    # We can iterate: r_i = min(r_i, min_j (d_ij - r_j))
    # This might decrease radii too much if not careful, but converges to a valid solution.
    # Better: use a solver or just trust the uniform radius found.
    
    # Actually, the optimization pushed circles apart to fit radius r_target.
    # So r_target should be feasible.
    # Let's verify and adjust.
    
    # Let's just set all radii to the largest feasible uniform radius found.
    # But maybe we can have unequal radii?
    # The problem asks to maximize SUM.
    # If we have a valid configuration with uniform radius r, sum is 26*r.
    # If we can make some larger and some smaller, sum might increase?
    # Unlikely to increase sum significantly if we are constrained by packing density.
    # But let's try to compute exact max radii.
    
    # Solve for max radii using a simple iterative relaxation (Bellman-Ford like)
    # Constraints: r_i <= b_i, r_i + r_j <= d_ij
    # Maximize sum r_i.
    # This is dual to min cost flow? No.
    # But we can just use the uniform radius r_opt found by simulation.
    
    # Calculate min distance between any pair
    min_pair_dist = float('inf')
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            d = np.linalg.norm(current_centers[i] - current_centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
                
    # Calculate min distance to boundary
    min_boundary_dist = float('inf')
    for i in range(n_circles):
        x, y = current_centers[i]
        d = min(x, 1-x, y, 1-y)
        if d < min_boundary_dist:
            min_boundary_dist = d
            
    # The max uniform radius is limited by min_pair_dist/2 and min_boundary_dist
    r_uniform = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # However, to be safe and strictly valid, we should compute radii properly.
    # Let's set radii to r_uniform.
    radii = np.full(n_circles, r_uniform)
    
    # Check validity and sum
    sum_radii = np.sum(radii)
    
    # Just in case, if we can squeeze more by making some smaller?
    # No, making smaller doesn't help sum.
    # Making larger is constrained by neighbors.
    # If we reduce a neighbor's radius, we can increase this one's?
    # r_i + r_j <= d. If we decrease r_j, r_i can increase.
    # But sum r_i + r_j is constant (bounded by d).
    # So redistributing radius doesn't change sum of pair, unless there are chain constraints.
    # But usually sum is maximized when radii are equal or balanced.
    
    # Let's double check if we can improve by solving the LP for radii.
    # But for now, uniform is a very good approximation.
    
    # Let's refine the radii calculation to be robust.
    # We can try to solve r_i + r_j <= d_ij, r_i <= b_i.
    # Initialize r_i = b_i.
    # Iterate: r_i = min(r_i, min_j (d_ij - r_j))
    # This finds the maximum r_i such that r_i + r_j <= d_ij for all j.
    # Wait, if we decrease r_j, the bound for r_i increases.
    # This simple iteration decreases radii.
    # Actually, the constraints define a polytope. We want to maximize sum.
    # This is a linear program.
    # Since N=26 is small, we could theoretically solve it, but it's complex to code from scratch without scipy.optimize.linprog.
    # But wait, scipy is allowed.
    
    try:
        from scipy.optimize import linprog
        
        # Variables: r_0, ..., r_25 (26 vars)
        # Objective: minimize -sum(r_i)  =>  maximize sum(r_i)
        c_obj = -np.ones(n_circles)
        
        # Constraints:
        # 1. r_i <= b_i  =>  r_i <= b_i  =>  [I] r <= b
        # 2. r_i + r_j <= d_ij  =>  [A] r <= d
        
        # Inequality constraints: A_ub * r <= b_ub
        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= b_i
        # We can add these as rows in A_ub: [0..1..0] r <= b_i
        # Or simply bounds. linprog supports bounds.
        bounds = []
        for i in range(n_circles):
            x, y = current_centers[i]
            b_val = min(x, 1-x, y, 1-y)
            bounds.append((0, b_val))
            
        # Pairwise constraints: r_i + r_j <= d_ij
        # This is many constraints: 26*25/2 = 325 constraints.
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                d_ij = np.linalg.norm(current_centers[i] - current_centers[j])
                # Row for r_i + r_j <= d_ij
                row = np.zeros(n_circles)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d_ij)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            radii = res.x
            sum_radii = np.sum(radii)
        else:
            # Fallback to uniform
            radii = np.full(n_circles, r_uniform)
            sum_radii = np.sum(radii)
            
    except ImportError:
        # Fallback if scipy not available or linprog fails
        radii = np.full(n_circles, r_uniform)
        sum_radii = np.sum(radii)

    return current_centers, radii, sum_radii
