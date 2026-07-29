# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1057391e) state=a6fac4a5 sum of radii=2.541421 correctness=1.0
# stdout(first 200): Optimizing configuration 1...   New best sum: 2.5414 Optimizing configuration 2... Optimizing configuration 3...
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import scipy.spatial as spatial

def solve_radii_lp(centers):
    """
    Given fixed centers, solve LP to find radii that maximize sum(radii)
    subject to non-overlap and boundary constraints.
    """
    n = len(centers)
    
    # Variables: r_0, ..., r_{n-1}
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # r_i <= x_i  => r_i <= x_i
    # r_i <= 1-x_i => r_i <= 1 - x_i
    # etc.
    # We can also model these as upper bounds on variables (bounds parameter in linprog)
    # But let's stick to standard form or use bounds.
    # Actually, linprog supports bounds directly.
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    # This is r_i + r_j <= d_ij
    # 1*r_i + 1*r_j <= d_ij
    
    # We will construct A_ub for pairwise constraints.
    # Number of pairs is n*(n-1)/2.
    # For n=26, ~325 constraints. This is small for LP.
    
    # Precompute distances
    dists = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Build A_ub and b_ub for pairwise constraints
    # We need a matrix where each row corresponds to a pair (i, j)
    # Row has 1 at col i, 1 at col j, 0 elsewhere.
    
    # Efficient construction
    A_ub_pairs = np.zeros((n * (n - 1) // 2, n))
    b_ub_pairs = np.zeros(n * (n - 1) // 2)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub_pairs[idx, i] = 1
            A_ub_pairs[idx, j] = 1
            b_ub_pairs[idx] = dists[i, j]
            idx += 1
            
    # Bounds for variables r_i: (0, infinity)
    # But also limited by boundaries.
    # We can include boundary constraints in bounds or A_ub.
    # Using bounds is cleaner.
    # r_i >= 0
    # r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        # Clamp max_r to be non-negative
        if max_r < 0:
            max_r = 0
        bounds_r.append((0, max_r))
        
    # Solve LP
    # Method 'highs' is recommended if available, else 'interior-point'
    try:
        res = opt.linprog(c, A_ub=A_ub_pairs, b_ub=b_ub_pairs, bounds=bounds_r, method='highs')
    except Exception:
        # Fallback if highs not available
        res = opt.linprog(c, A_ub=A_ub_pairs, b_ub=b_ub_pairs, bounds=bounds_r, method='interior-point')
        
    if res.success:
        return res.x, -res.fun
    else:
        # If LP fails, return small radii to avoid crash, but this shouldn't happen often
        # unless centers are impossible (e.g. distance 0)
        # In case of dist 0, bounds might conflict? No, dist 0 implies r_i+r_j <= 0 => r_i=r_j=0.
        return np.zeros(n), 0.0

def objective_function(centers_flat):
    """
    Objective function for optimizing centers.
    Inputs: flattened array of centers (x1, y1, x2, y2, ...)
    Returns: negative sum of radii (to minimize)
    """
    n = len(centers_flat) // 2
    centers = centers_flat.reshape((n, 2))
    
    # Solve for optimal radii
    radii, sum_r = solve_radii_lp(centers)
    
    # Return negative sum for minimization
    return -sum_r

def run_packing():
    """
    Main function to pack 26 circles in a unit square.
    """
    n_circles = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Helper to evaluate a configuration
    def evaluate(centers):
        radii, s = solve_radii_lp(centers)
        return s, radii

    # Initialization 1: Hexagonal Grid
    # Try to pack 26 points in a hexagonal lattice pattern
    # We can try a few variations of rows/cols
    configs = []
    
    # Variation 1: 6 rows, staggered
    # Estimate row count and spacing
    # Try to fit points roughly
    # Let's create a hexagonal grid and trim/pad to 26
    # Spacing dx = 0.2, dy = 0.2 * sqrt(3) / 2 approx? 
    # Let's just generate a dense hexagonal grid and pick 26 best or first 26
    # Or just use a perturbed grid.
    
    # Heuristic: Start with a 5x5 grid (25 points) + 1 point in center gap
    # 5x5 grid points: (0.1 + 0.2*i, 0.1 + 0.2*j) for i,j in 0..4
    grid_centers = []
    for i in range(5):
        for j in range(5):
            grid_centers.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    # Add one in center? Center of square is 0.5, 0.5. 
    # But 0.5, 0.5 is already occupied? 
    # 5x5 grid covers 0.1, 0.3, 0.5, 0.7, 0.9. So (0.5, 0.5) is occupied.
    # Let's place 26th near a gap. Gaps are at (0.2, 0.2) etc.
    # Add at (0.2, 0.2)
    grid_centers.append([0.2, 0.2])
    
    # Trim to 26
    configs.append(np.array(grid_centers[:26]))
    
    # Variation 2: Perturbed Hexagonal
    # Generate points in hex pattern
    hex_centers = []
    # Try rows y = 0.1, 0.25, 0.4, 0.55, 0.7, 0.85 ?
    # Hex spacing: dy = sqrt(3)/2 * dx.
    # Let's try dx = 0.18, dy = 0.155
    # Just generate random-ish hex grid
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    # Start y at dy/2? No, center at r.
    # Let's just scatter
    for row in range(6):
        y = 0.1 + row * dy
        if y > 0.9: break
        # x offset for even/odd rows
        x_start = 0.1 if row % 2 == 0 else 0.1 + dx/2
        col = 0
        while True:
            x = x_start + col * dx
            if x > 0.9: break
            hex_centers.append([x, y])
            col += 1
        if len(hex_centers) >= 26:
            break
            
    if len(hex_centers) < 26:
        # Pad with random or grid points
        while len(hex_centers) < 26:
            hex_centers.append([0.5, 0.5]) # Placeholder
    configs.append(np.array(hex_centers[:26]))
    
    # Variation 3: Random perturbation of grid
    grid_pert = configs[0].copy()
    grid_pert += np.random.normal(0, 0.02, grid_pert.shape)
    configs.append(grid_pert)

    # Run optimization on each config
    for k, initial_centers in enumerate(configs):
        print(f"Optimizing configuration {k+1}...")
        
        # Flatten centers
        x0 = initial_centers.flatten()
        
        # Bounds for centers: [0, 1] for each coordinate
        bounds_centers = [(0, 1)] * (n_circles * 2)
        
        # Optimization
        # Nelder-Mead is good for non-smooth objectives (LP result is continuous but kinks)
        # Maxiter might need to be high
        res = opt.minimize(
            objective_function,
            x0,
            method='Nelder-Mead',
            bounds=bounds_centers,
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
        )
        
        if res.success or res.fun < -best_sum: # We minimize -sum, so smaller is better
            # Evaluate final result
            final_centers = res.x.reshape((n_circles, 2))
            # Clip centers to [0, 1] strictly
            final_centers = np.clip(final_centers, 0, 1)
            
            s, r = evaluate(final_centers)
            if s > best_sum:
                best_sum = s
                best_centers = final_centers
                best_radii = r
                print(f"  New best sum: {s:.4f}")

    # If we didn't find anything good (should not happen), fallback to grid
    if best_sum == 0.0:
        s, r = evaluate(configs[0])
        best_sum = s
        best_centers = configs[0]
        best_radii = r

    # Final validation and cleanup
    # Ensure radii are not NaN and centers valid
    if np.any(np.isnan(best_radii)) or np.any(np.isnan(best_centers)):
        print("Warning: NaN detected, falling back to simple grid")
        # Simple fallback
        centers = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.2, 0.2]])
        radii, best_sum = evaluate(centers)
        best_centers = centers
        best_radii = radii

    # Sort circles by index (optional, but good for consistency)
    # The problem doesn't require specific ordering.
    
    return best_centers, best_radii, float(best_sum)

# Note: The function run_packing must be defined. 
# The code above defines it. 
# We need to make sure imports are available.
# The environment usually has numpy and scipy.
