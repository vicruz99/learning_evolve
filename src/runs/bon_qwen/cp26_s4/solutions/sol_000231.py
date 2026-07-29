# sol_000231 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2fe8b400) state=5b4e1ce9 sum of radii=2.417882 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def compute_max_radii(centers):
    """
    Computes the maximal radii for a given set of centers such that
    circles do not overlap and stay within the unit square.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Precompute distances to boundaries
    # r_i <= x_i, 1-x_i, y_i, 1-y_i
    bound_r = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                         np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    # Compute pairwise distances
    # We can compute this efficiently using broadcasting or loops
    # For N=26, loops are fine.
    
    for i in range(n):
        # Initialize with boundary constraint
        r = bound_r[i]
        
        # Check constraints against all other circles
        # r_i + r_j <= dist(i, j)  =>  r_i <= dist(i, j) - r_j
        # But r_j is also unknown.
        # However, the maximal valid radius for center i, assuming other circles
        # also take their maximal valid radii based on the SAME centers, 
        # is determined by the bottleneck.
        # Actually, the definition r_i = min(bound_i, 0.5 * min_j dist(i,j)) 
        # ensures a valid packing.
        # Is it the maximal sum? 
        # If we set r_i = 0.5 * min_j dist(i,j), then r_i + r_j <= 0.5 d_ij + 0.5 d_ji = d_ij.
        # So this assignment is always valid.
        # Is it maximal? Yes, because r_i cannot exceed 0.5 * dist(i,j) for any j.
        
        min_dist = 1.0 # Large number
        xi, yi = centers[i]
        
        # Check distance to all other centers
        # Vectorized approach for speed if needed, but loop is safe
        diffs = centers - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Ignore distance to self (index i)
        # dists[i] is 0
        others = np.delete(dists, i)
        
        if len(others) > 0:
            min_dist_to_other = np.min(others)
            r = min(r, 0.5 * min_dist_to_other)
            
        radii[i] = r
        
    return radii

def objective_func(vars, n):
    """
    Objective function to minimize: -sum(radii)
    vars is a flattened array of centers: [x1, y1, x2, y2, ..., xn, yn]
    """
    centers = vars.reshape((n, 2))
    
    # Clip centers to [0, 1] to prevent invalid states during optimization
    # Although optimizer should handle bounds, clipping helps stability
    # But strictly, centers must be in [0,1]. 
    # Nelder-Mead doesn't support bounds easily, so we map or clip?
    # Better to use bounds in minimize if possible, or just rely on gradient.
    # Let's just compute radii. If center is outside, radius might be negative or logic fails.
    # Let's enforce bounds by penalizing or clipping.
    
    # A robust way for boundless optimizers:
    # Map variables to [0,1] using sigmoid or just clamp?
    # Clamping stops movement at boundaries.
    
    centers_clipped = np.clip(centers, 0.0, 1.0)
    
    radii = compute_max_radii(centers_clipped)
    return -np.sum(radii)

def get_initial_centers(n):
    """
    Generate initial centers using a hexagonal lattice pattern 
    that fits n points into the unit square.
    """
    centers = []
    # Estimate spacing
    # Area per point approx 1/n. Side approx 1/sqrt(n).
    # Hexagonal packing density factor.
    # Let's just try to fit a grid and perturb.
    
    # Strategy: Create a grid of potential points and pick best n?
    # Or just generate a hex lattice and select points that fit?
    
    # Simple approach: Random points with repulsion (force directed)
    # Or just a dense grid and select a subset?
    # For 26, a 6x5 grid has 30 points. We can remove 4.
    # But we want specific 26 points.
    
    # Let's try a hexagonal arrangement directly.
    # Rows
    # Spacing y = sqrt(3)/2 * x_spacing.
    # Let's optimize the scaling factor.
    
    # Try to fit as many as possible with r=0.1 (d=0.2)
    # 5x5 grid is 25.
    # Maybe a 6x5 grid with some points removed?
    
    # Let's generate a set of candidates on a hex grid and run optimization.
    # To be safe, let's start with a random perturbation of a grid.
    
    rows = 6
    cols = 5
    points = []
    
    # Hex grid generation
    # Row i, Col j
    # x = j * 2*r + (i%2)*r + offset_x
    # y = i * sqrt(3)*r + offset_y
    
    # We don't know optimal r. Let's assume r=0.1 for layout.
    r_est = 0.1
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Center the grid in [0,1]
    # Width needed: (cols-1)*dx + 2*r_est (for radius padding) ?
    # Actually centers should be in [r, 1-r].
    # Let's just place centers in [0,1] and let optimizer fix.
    
    # Generate a dense grid of points
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            points.append([x, y])
    
    # Normalize to fit in [0, 0.9] roughly
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Scale to fit in [0.1, 0.9]
        width = max_x - min_x
        height = max_y - min_y
        
        target_w = 0.8
        target_h = 0.8
        
        scale_x = target_w / width if width > 0 else 1
        scale_y = target_h / height if height > 0 else 1
        
        scaled_points = []
        for p in points:
            nx = (p[0] - min_x) * scale_x + 0.1
            ny = (p[1] - min_y) * scale_y + 0.1
            scaled_points.append([nx, ny])
        
        # We need exactly n=26 points.
        # We generated 6*5 = 30 points.
        # Select 26 points.
        # Which ones to drop?
        # Dropping corners might be bad. Dropping edges?
        # Maybe just take the first 26?
        # Or use a selection that keeps density uniform.
        
        # Better: Use a 5x6 grid (30) and remove 4 points from corners/edges?
        # Actually, optimization will move them.
        # Let's just take the first 26 of the scaled points?
        # The order is row by row.
        # It creates a block.
        
        candidates = scaled_points[:n]
        return np.array(candidates)
    
    # Fallback: Random
    return np.random.rand(n, 2)

def run_packing():
    n = 26
    
    # Strategy: Run optimization from multiple starting points
    best_sum = -1.0
    best_centers = None
    
    # Starting configurations
    # 1. Hexagonal grid based
    # 2. Random perturbation of grid
    # 3. Pure random (seeded)
    
    start_configs = []
    
    # Config 1: Hex-like grid
    start_configs.append(get_initial_centers(n))
    
    # Config 2: Grid 5x5 + 1 center
    grid = np.array([[i/4.5 + 1/9, j/4.5 + 1/9] for i in range(5) for j in range(5)])
    # Add center
    center = np.array([[0.5, 0.5]])
    combined = np.vstack([grid, center])
    # We have 26 points. But they are not well distributed (clumped in 5x5).
    # The 5x5 grid uses [0.11, 0.88].
    # This is a valid start.
    start_configs.append(combined[:n])
    
    # Config 3: Random with large repulsion (simulated briefly)
    np.random.seed(42)
    rand_centers = np.random.rand(n, 2)
    # Simple repulsion step
    for _ in range(100):
        forces = np.zeros_like(rand_centers)
        for i in range(n):
            for j in range(i+1, n):
                diff = rand_centers[i] - rand_centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    force = diff / (dist**2) # Repulsive
                    forces[i] += force
                    forces[j] -= force
            # Boundary repulsion
            x, y = rand_centers[i]
            if x < 0.1: forces[i, 0] += (0.1 - x)
            if x > 0.9: forces[i, 0] -= (x - 0.9)
            if y < 0.1: forces[i, 1] += (0.1 - y)
            if y > 0.9: forces[i, 1] -= (y - 0.9)
        
        rand_centers += 0.05 * forces
        rand_centers = np.clip(rand_centers, 0.01, 0.99)
    start_configs.append(rand_centers)
    
    for idx, start_centers in enumerate(start_configs):
        # Flatten
        x0 = start_centers.flatten()
        
        # Optimization
        # Nelder-Mead
        res = minimize(
            objective_func, 
            x0, 
            args=(n,), 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if res.success or res.nit > 0:
            centers_opt = res.x.reshape((n, 2))
            # Clip just in case
            centers_opt = np.clip(centers_opt, 1e-5, 1-1e-5)
            
            radii = compute_max_radii(centers_opt)
            current_sum = np.sum(radii)
            
            # Validate roughly
            # Check for NaN or negative
            if not np.isnan(radii).any() and np.all(radii >= 0):
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers_opt.copy()
    
    # Final calculation
    if best_centers is not None:
        final_radii = compute_max_radii(best_centers)
        return best_centers, final_radii, np.sum(final_radii)
    else:
        # Fallback
        return np.zeros((n, 2)), np.zeros(n), 0.0
