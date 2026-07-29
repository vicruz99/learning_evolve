# sol_000277 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 085da352) state=5b2da40b sum of radii=2.609646 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_hexagonal_centers(n):
    """
    Generates centers for n circles in a hexagonal packing pattern.
    Returns an (n, 2) array of centers.
    """
    centers = []
    # Approximate number of circles per row
    # Hexagonal packing density allows approx sqrt(n) * 0.6 per dimension?
    # Let's just try to fit them.
    # Estimate rows
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    
    # Try to distribute n circles into rows
    # A simple distribution
    cols = int(np.ceil(n / rows))
    
    # Refine rows based on area
    # Area of one hex cell is sqrt(3)/2 * d^2? No.
    # Let's just generate points on a grid and trim/fill.
    
    # Better approach: Generate a dense hex grid and pick the first n
    # or place them intelligently.
    
    # Let's try a rectangular grid perturbation as fallback, 
    # but hex is better.
    # Let's construct rows.
    # Row height = sqrt(3)/2 * width
    # Let's assume width is approx 1/sqrt(n) ?
    
    # Heuristic for 26:
    # Maybe 6 rows? 5, 4, 5, 4, 5, 3?
    # Let's try to fit into 1x1.
    
    y_current = 0.1 # Start with margin?
    # Actually let's just place them and scale.
    
    points = []
    row_y = 0
    row_idx = 0
    while len(points) < n:
        # Number of circles in this row
        # Alternate between ceil and floor of sqrt(n)?
        # Or just fill.
        # Let's assume width 1.0.
        # Spacing dx.
        # We want to pack roughly n circles.
        
        # Let's try a standard hex grid generation
        # y step = sqrt(3)/2 * x_step
        # Let's pick a target spacing.
        # For 26 circles, average radius ~ 0.1. Diameter 0.2.
        # Spacing ~ 0.2.
        dx = 0.2
        dy = dx * np.sqrt(3) / 2
        
        # x positions for this row
        # Even rows (0, 2...): 0, dx, 2dx...
        # Odd rows (1, 3...): dx/2, 3dx/2...
        
        offset = (dx / 2) if (row_idx % 2 == 1) else 0
        
        x_current = offset
        while x_current <= 1.0 - 0.1: # 0.1 margin roughly
            points.append([x_current, row_y])
            x_current += dx
            if len(points) >= n:
                break
        
        row_y += dy
        row_idx += 1
        
    # We might have generated too many or too few, or out of bounds.
    # Let's just take the first n valid points in [0,1]x[0,1]
    valid_points = []
    for p in points:
        if 0 <= p[0] <= 1 and 0 <= p[1] <= 1:
            valid_points.append(p)
    
    if len(valid_points) < n:
        # Fallback to random
        valid_points = list(np.random.rand(n, 2))
    
    return np.array(valid_points[:n])

def solve_lp_radii(centers):
    """
    Given centers, solve LP to maximize sum of radii.
    Returns (radii, sum_radii).
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints:
    # 1. r_i >= 0 (handled by bounds)
    # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # 3. r_i + r_j <= dist(i, j)
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    bounds = []
    A_ub = []
    b_ub = []
    
    # Precompute distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            dists[i, j] = d
            dists[j, i] = d
            
    # Add distance constraints
    # r_i + r_j <= d_ij
    # Row for (i, j): 1 at i, 1 at j
    constraint_rows = []
    constraint_rhs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraint_rows.append(row)
            constraint_rhs.append(dists[i, j])
            
    # Add boundary constraints
    # r_i <= b_i
    # This can be added to A_ub or handled via bounds?
    # If we use bounds, we can't easily get the "tightness" for force calculation logic later,
    # but for the LP itself, bounds are fine.
    # However, for consistency with force logic (checking active constraints),
    # let's put them in A_ub so we can inspect slacks? 
    # Actually linprog doesn't easily expose duals for bounds in all versions.
    # But we can just check the values of r against bounds after solving.
    
    # Let's use bounds for simplicity in LP formulation.
    for i in range(n):
        x, y = centers[i]
        b_val = min(x, 1 - x, y, 1 - y)
        # Ensure non-negative bound
        b_val = max(0.0, b_val)
        bounds.append((0, b_val))
        
    # Convert list to matrix
    if constraint_rows:
        A_ub = np.array(constraint_rows)
        b_ub = np.array(constraint_rhs)
    else:
        A_ub = np.array([])
        b_ub = np.array([])
        
    # Solve LP
    # method='highs' is robust and fast
    try:
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, np.sum(res.x)
        else:
            # Fallback if LP fails (should not happen)
            # Return small radii
            small_r = np.full(n, 1e-6)
            return small_r, np.sum(small_r)
    except Exception:
        return np.full(n, 1e-6), 0.0

def run_packing():
    """
    Optimizes packing of 26 circles.
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Strategy: Run optimization loop multiple times with different seeds/initializations
    # But to keep it fast, maybe just one strong run with good init.
    # Let's try 3 restarts.
    
    restarts = 5
    iterations_per_restart = 300
    
    for restart in range(restarts):
        # Initialize centers
        if restart == 0:
            # Try hexagonal grid
            centers = get_hexagonal_centers(n)
        else:
            # Random perturbation of best found so far or random
            if best_centers is not None:
                centers = best_centers + np.random.normal(0, 0.01, size=(n, 2))
            else:
                centers = np.random.rand(n, 2)
        
        # Clamp to [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Optimization parameters
        step_size = 0.05
        
        local_best_sum = -1.0
        local_best_centers = None
        local_best_radii = None
        
        # We need to keep a copy of centers to not destroy best state
        # But here we work on 'centers' variable
        
        for it in range(iterations_per_restart):
            # 1. Solve LP for radii
            radii, current_sum = solve_lp_radii(centers)
            
            if current_sum > local_best_sum:
                local_best_sum = current_sum
                local_best_centers = centers.copy()
                local_best_radii = radii.copy()
            
            # 2. Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion if touching
            # Check constraints r_i + r_j <= dist
            # If r_i + r_j approx dist, push apart
            tol = 1e-5
            
            # We need distances again
            # Recompute or store? Recomputing is cheap for n=26
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    dists[i, j] = d
                    
            for i in range(n):
                # Boundary forces
                x, y = centers[i]
                r = radii[i]
                
                # Left wall
                if r >= x - tol:
                    forces[i, 0] += 1.0
                # Right wall
                if r >= (1 - x) - tol:
                    forces[i, 0] -= 1.0
                # Bottom wall
                if r >= y - tol:
                    forces[i, 1] += 1.0
                # Top wall
                if r >= (1 - y) - tol:
                    forces[i, 1] -= 1.0
                    
                # Neighbor repulsion
                for j in range(i + 1, n):
                    d = dists[i, j]
                    # Avoid div by zero
                    if d < 1e-9:
                        d = 1e-9
                        # Random push if exactly overlapping
                        forces[i] += np.random.randn(2) * 0.1
                        forces[j] -= np.random.randn(2) * 0.1
                    else:
                        if (radii[i] + radii[j]) >= d - tol:
                            # Active constraint, repel
                            # Vector from j to i
                            vec = centers[i] - centers[j]
                            # Normalize
                            vec /= d
                            # Force on i is +vec (away from j)
                            # Force on j is -vec
                            forces[i] += vec
                            forces[j] -= vec
            
            # 3. Update centers
            # Apply force
            # Adaptive step size? Decay.
            alpha = step_size * (1.0 - it / iterations_per_restart)
            alpha = max(alpha, 0.001)
            
            centers += alpha * forces
            
            # Clamp
            centers = np.clip(centers, 0.0, 1.0)
            
        # Update global best
        if local_best_sum > best_sum:
            best_sum = local_best_sum
            best_centers = local_best_centers
            best_radii = local_best_radii

    # Final verification and adjustment
    # Solve LP one last time for the best centers to ensure radii are optimal
    final_radii, final_sum = solve_lp_radii(best_centers)
    
    # Ensure non-negative radii (LP should guarantee, but just in case)
    final_radii = np.maximum(final_radii, 0.0)
    
    # Recompute sum
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum

if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    print(f"Number of circles: {len(radii)}")
    
    # Quick self-validation
    # Check bounds
    valid = True
    for i in range(len(radii)):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            print(f"Circle {i} out of bounds")
            valid = False
        if r < 0:
            print(f"Circle {i} negative radius")
            valid = False
            
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-9:
                print(f"Circles {i} and {j} overlap: dist={dist}, sum_r={radii[i]+radii[j]}")
                valid = False
                
    if valid:
        print("Packing is valid.")
    else:
        print("Packing is INVALID.")
