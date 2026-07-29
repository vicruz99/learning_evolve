# sol_000073 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 25735fc7) state=b17672b7 sum of radii=2.139449 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import random

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

def run_packing():
    n_circles = 26
    
    # Initialize centers in a perturbed hexagonal lattice
    # This gives a better starting density than a square grid
    centers = np.zeros((n_circles, 2))
    
    # Try to fit rows of hexagonal packing
    # Approximate radius 0.105 (based on area heuristic)
    # Rows of 6, 5, 6, 5, 4? Total 26.
    # Let's just scatter them densely and let the optimizer work.
    
    # Simple grid initialization with some randomness
    # 6x5 grid roughly
    # We want to cover [0,1]x[0,1]
    x_coords = np.linspace(0.1, 0.9, 6)
    y_coords = np.linspace(0.1, 0.9, 5)
    
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx < n_circles:
                # Add small noise to break symmetry
                noise_x = (random.random() - 0.5) * 0.02
                noise_y = (random.random() - 0.5) * 0.02
                centers[idx, 0] = max(0.01, min(0.99, x_coords[j] + noise_x))
                centers[idx, 1] = max(0.01, min(0.99, y_coords[i] + noise_y))
                idx += 1
    
    # Initial radii estimate
    radii = np.ones(n_circles) * 0.08

    # Optimization loop
    # We will use a force-directed relaxation approach
    # Forces push circles apart if they are too close relative to their radii
    # And push radii up
    
    # Parameters for relaxation
    max_iters = 2000
    dt = 0.05 # Step size for center movement
    lr_r = 0.1 # Learning rate for radii
    
    # Precompute index pairs for overlap check
    pairs = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            pairs.append((i, j))
            
    best_sum_radii = 0
    best_centers = centers.copy()
    best_radii = radii.copy()

    for step in range(max_iters):
        # 1. Update Radii based on current centers
        # We want to maximize sum(r) subject to r_i + r_j <= dist(i,j) and r_i <= dist(i, boundary)
        # This is a Linear Programming problem, but we can approximate with simple relaxation
        
        # Calculate distance to boundaries
        dist_boundary = np.minimum(
            np.minimum(centers[:, 0], 1 - centers[:, 0]),
            np.minimum(centers[:, 1], 1 - centers[:, 1])
        )
        
        # Calculate distances between centers
        # Efficiently compute pairwise distances
        # centers shape (N, 2)
        # diffs shape (N, N, 2)
        # dists shape (N, N)
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf) # Ignore self
        
        # Current constraints: r_i + r_j <= dists[i,j]
        # r_i <= dist_boundary[i]
        
        # We can iteratively adjust radii
        # r_i_new = min(dist_boundary[i], min_j( (dists[i,j] - r_j + r_i) / 2 )) ?
        # Actually, a simple projection:
        # r_i = min(dist_boundary[i], 0.5 * min(dists[i, :])) is a valid lower bound solution, 
        # but maybe not optimal sum.
        # However, for equal radii, it is optimal. For unequal, it's a good heuristic.
        
        # Let's use a simple gradient ascent on radii with constraints
        # grad_r_i = 1 - sum_{j} lambda_{ij} - lambda_{boundary, i}
        # But easier: just enforce constraints and push up.
        
        for _ in range(5): # Sub-iterations for radii
            for i in range(n_circles):
                # Constraint from boundary
                max_r_boundary = dist_boundary[i]
                
                # Constraint from neighbors
                # r_i <= dists[i, j] - r_j  => r_i + r_j <= dists[i, j]
                # We want to increase r_i.
                # If r_i + r_j > dists[i, j], we must reduce.
                # If r_i + r_j < dists[i, j], we can increase r_i or r_j.
                
                # Heuristic: set r_i to be as large as possible respecting neighbors
                # r_i <= (dists[i, j] - r_j) for all j? No, that's too conservative.
                # We just need r_i + r_j <= dists[i, j].
                
                # Let's just clamp radii to satisfy constraints, then scale up?
                pass
            
            # A robust way to solve for optimal radii given fixed centers is LP, 
            # but let's use a simple iterative fix:
            # r_i = min(dist_boundary[i], 0.5 * min_j(dists[i, j] + r_i - r_j)) ??
            # Actually, if we just enforce r_i + r_j <= dist, we can do:
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    d = dists[i, j]
                    if radii[i] + radii[j] > d:
                        # Overlap. Reduce radii equally or proportionally?
                        # To preserve sum, maybe just shrink.
                        # But we want to maximize sum.
                        # This implies we are currently in an invalid state or suboptimal.
                        # Let's just enforce validity first.
                        diff = radii[i] + radii[j] - d
                        radii[i] -= diff / 2
                        radii[j] -= diff / 2
                        radii[i] = max(0, radii[i])
                        radii[j] = max(0, radii[j])
            
            # Clamp to boundary
            radii = np.minimum(radii, dist_boundary)
            radii = np.maximum(radii, 0)

        # 2. Update Centers using forces
        # Force on circle i:
        # Repulsion from other circles if dist < r_i + r_j + margin
        # Attraction to boundaries? No, boundaries repel.
        # We want to spread centers to allow larger radii.
        
        forces = np.zeros_like(centers)
        
        # Calculate gradients of the "constraint violation" or "potential"
        # Potential function: sum over pairs of max(0, r_i + r_j - dist)^2
        # We want to minimize this potential (make it 0) while keeping radii high.
        # Actually, we want to move centers to increase the "capacity" for radii.
        # Capacity for r_i is determined by min(dist_boundary, min_j(dists[i,j] - r_j)).
        # So we want to increase dists[i,j].
        
        # Simple repulsive force:
        # If dists[i, j] < r_i + r_j + epsilon, push apart.
        # Force magnitude proportional to overlap.
        
        # Also, we can try to move centers to "open up" space.
        # If a circle is constrained by a neighbor, move away from it.
        
        for i in range(n_circles):
            fx, fy = 0.0, 0.0
            
            # Force from boundaries
            # If r_i is close to boundary limit, push away from boundary
            # x - r >= 0 -> if x - r is small, push +x
            # But this is coupled with r.
            # Let's assume we want to center circles more if possible?
            # Actually, corners are good for large circles?
            # Let's just use repulsion.
            
            for j in range(n_circles):
                if i == j: continue
                
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                    dx, dy = 0.01, 0.01
                
                r_sum = radii[i] + radii[j]
                
                # Repulsion force
                # We want dist >= r_sum.
                # If dist < r_sum, push hard.
                # If dist > r_sum, maybe a weak attraction to keep them close? 
                # No, we want to maximize radii, so spreading out is good.
                # But we are bounded by the square.
                
                # Let's use a force that pushes apart if they are "tight"
                # Tightness = r_sum - dist
                tightness = r_sum - dist
                
                # Add a small margin to ensure separation
                margin = 0.01 
                effective_tightness = max(0, tightness - margin)
                
                force_mag = effective_tightness * 10.0 # Stiffness
                
                if dist > 0:
                    fx += (dx / dist) * force_mag
                    fy += (dy / dist) * force_mag
            
            # Boundary repulsion
            # If circle is touching boundary, push in
            # But we want to allow touching.
            # However, if it's "squeezed", maybe move to center?
            # Let's add a weak force towards center to prevent clustering at edges unnecessarily?
            # Or weak force away from boundaries?
            # Actually, circles at boundaries are fine.
            # But if they are stuck in corner, maybe they can't grow.
            
            # Weak repulsion from boundaries
            boundary_margin = 0.02
            if centers[i, 0] < boundary_margin:
                fx += 1.0
            if centers[i, 0] > 1 - boundary_margin:
                fx -= 1.0
            if centers[i, 1] < boundary_margin:
                fy += 1.0
            if centers[i, 1] > 1 - boundary_margin:
                fy -= 1.0
                
            forces[i, 0] = fx
            forces[i, 1] = fy
            
        # Update centers
        # Learning rate decay
        current_dt = dt * (1.0 - step / max_iters)
        centers += forces * current_dt * 0.1
        
        # Keep centers in bounds [r, 1-r] is hard as r changes.
        # Just keep in [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # Update radii again after moving centers to reflect new capacity
        # Re-apply constraints
        for _ in range(2):
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    if radii[i] + radii[j] > d:
                        diff = radii[i] + radii[j] - d
                        radii[i] -= diff / 2
                        radii[j] -= diff / 2
            radii = np.minimum(radii, np.minimum(centers[:, 0], 1 - centers[:, 0]))
            radii = np.minimum(radii, np.minimum(centers[:, 1], 1 - centers[:, 1]))
            radii = np.maximum(radii, 0)

        # Track best
        current_sum = np.sum(radii)
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
            # Optional: Print progress
            # print(f"Step {step}, Sum Radii: {current_sum:.4f}")

    # Final cleanup and validation
    # Ensure strict validity
    # Run a few more aggressive constraint fixes
    for _ in range(100):
        changed = False
        # Fix overlaps
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < best_radii[i] + best_radii[j]:
                    # Reduce radii to just touch
                    sum_r = best_radii[i] + best_radii[j]
                    ratio = dist / sum_r
                    best_radii[i] *= ratio
                    best_radii[j] *= ratio
                    changed = True
        
        # Fix boundaries
        for i in range(n_circles):
            x, y = best_centers[i]
            r = best_radii[i]
            min_dist = min(x, 1-x, y, 1-y)
            if r > min_dist:
                best_radii[i] = min_dist
                changed = True
        if not changed:
            break
            
    return best_centers, best_radii, np.sum(best_radii)

# Note: The above simulation code is a bit generic. 
# To be more robust and reach the target, let's implement a more deterministic optimization 
# using scipy if possible, or a more structured heuristic.
# However, given the constraints and the need for a self-contained solution, 
# I will refine the initialization and the relaxation loop.

# Refined Strategy for the actual code block:
# 1. Use a perturbed hexagonal grid.
# 2. Use scipy.optimize.minimize to maximize sum of radii directly?
#    Variables: 26*2 centers + 26 radii = 78 vars.
#    Constraints: r_i >= 0, r_i + r_j <= dist, r_i <= dist_boundary.
#    This is hard for scipy due to non-convexity.
#    
# 3. Better: Use a "jiggle" method.
#    - Fix radii, optimize centers to maximize min separation (maximize clearance).
#    - Fix centers, solve for optimal radii (LP or simple projection).
#    - Iterate.

def run_packing():
    n = 26
    
    # 1. Initialize with a dense packing
    # Try to fit 26 circles in a hexagonal pattern
    # Rows: 6, 5, 6, 5, 4 -> 26 circles
    # Approximate radius r=0.105
    # Height of row spacing: r * sqrt(3) approx 0.18
    # 5 rows: height approx 4 * 0.18 + 2r = 0.72 + 0.21 = 0.93 (fits in 1)
    # Width of row with 6 circles: 6 * 2r = 1.26 (too wide!)
    # So we cannot fit 6 circles of radius 0.105 in width 1.
    # Max circles in width 1 with r=0.105 is floor(1 / 0.21) = 4.
    # So we need more rows.
    
    # Let's try 7 rows.
    # 7 rows of hex packing.
    # Vertical spacing r*sqrt(3).
    # Height = 2r + 6*r*sqrt(3) = r(2 + 10.39) = 12.39r <= 1 => r <= 0.08.
    # This reduces radius too much.
    
    # Alternative: Distorted packing.
    # Just place centers in a grid and let the optimizer find the configuration.
    # A 6x5 grid (30 points) -> remove 4? Or just 26 points.
    # 5 rows, 6 cols? 30 points.
    # Let's place 26 points randomly but clustered to fill space.
    
    np.random.seed(42)
    centers = np.random.rand(n, 2)
    radii = np.ones(n) * 0.05
    
    # Iterative Optimization
    # We will try to maximize sum(r) by relaxing constraints.
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = 0.0
    
    # We will perform multiple restarts with different initializations
    for restart in range(5):
        if restart == 0:
            # Hexagonal perturbation
            centers = np.zeros((n, 2))
            idx = 0
            # Try a pattern that fits well
            # 5 rows, alternating 6 and 5 circles? No, width limit.
            # Maybe 5 rows of 5 circles (25) + 1 circle?
            # 5x5 grid of radius 0.1 fits exactly.
            # Let's start with 5x5 grid + 1 extra circle in the center?
            # 5x5 grid centers:
            x_grid = np.linspace(0.1, 0.9, 5)
            y_grid = np.linspace(0.1, 0.9, 5)
            for i in range(5):
                for j in range(5):
                    if idx < n:
                        centers[idx] = [x_grid[j], y_grid[i]]
                        idx += 1
            # Last circle at center
            if idx < n:
                centers[idx] = [0.5, 0.5]
                idx += 1
            
            # Add noise
            centers += np.random.normal(0, 0.01, centers.shape)
            radii = np.ones(n) * 0.1
            
        elif restart == 1:
            # Random dense packing
            centers = np.random.rand(n, 2)
            radii = np.ones(n) * 0.08
            
        else:
            # Random
            centers = np.random.rand(n, 2)
            radii = np.ones(n) * 0.06

        # Optimization Loop
        # We use a simple force-directed method with dynamic radii
        for step in range(1000):
            # 1. Adjust Radii to satisfy constraints (Projection)
            # Maximize sum(r) subject to r_i + r_j <= dist(i,j) and r_i <= dist(i, wall)
            # This is an LP. We can solve it approximately.
            # Or use the "equal radius" projection: r_i = 0.5 * min(dist_to_neighbor, dist_to_wall)
            # But this is not optimal for sum.
            # Let's use a simple iterative solver for the radii.
            
            # Calculate distances
            dists_matrix = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2))
            np.fill_diagonal(dists_matrix, np.inf)
            
            dist_wall = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                                   np.minimum(centers[:, 1], 1 - centers[:, 1]))
            
            # Iterative projection for radii
            # r_i <= dist_wall[i]
            # r_i <= dists_matrix[i, j] - r_j
            # We can just enforce these.
            
            for _ in range(10): # Sub-iterations
                # Enforce boundary
                radii = np.minimum(radii, dist_wall)
                
                # Enforce pairwise
                for i in range(n):
                    for j in range(i+1, n):
                        d = dists_matrix[i, j]
                        if radii[i] + radii[j] > d:
                            # Reduce sum to d
                            current_sum = radii[i] + radii[j]
                            factor = d / current_sum
                            radii[i] *= factor
                            radii[j] *= factor
                
                radii = np.maximum(radii, 0)

            # 2. Move Centers to increase radii capacity
            # Gradient of sum(r) w.r.t centers?
            # Hard to compute directly.
            # Use repulsion: if r_i + r_j is close to dist(i,j), push apart.
            
            forces = np.zeros_like(centers)
            
            # We want to increase min(dist(i,j) - (r_i + r_j))
            # Let slack_ij = dist(i,j) - r_i - r_j
            # We want slack > 0.
            # If slack is small, we want to increase dist(i,j).
            # Force on i from j: + (slack_ij < 0 ? push : weak pull?)
            # Actually, if slack is positive, we can increase radii.
            # To maximize radii, we want to maximize slack.
            # So push apart regardless, but stronger when tight.
            
            # Also, walls repel.
            
            for i in range(n):
                fx, fy = 0.0, 0.0
                
                # Wall forces
                # If r_i is close to wall limit, push away
                # Limit is dist_wall[i].
                # If radii[i] >= dist_wall[i] - epsilon, push away from wall.
                
                # Left wall
                if centers[i, 0] - radii[i] < 0.01:
                    fx += 1.0
                # Right wall
                if centers[i, 0] + radii[i] > 0.99:
                    fx -= 1.0
                # Bottom wall
                if centers[i, 1] - radii[i] < 0.01:
                    fy += 1.0
                # Top wall
                if centers[i, 1] + radii[i] > 0.99:
                    fy -= 1.0
                
                # Neighbor forces
                for j in range(n):
                    if i == j: continue
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6: dist = 1e-6
                    
                    r_sum = radii[i] + radii[j]
                    
                    # We want dist >= r_sum.
                    # If dist < r_sum, strong repulsion.
                    # If dist > r_sum, maybe weak repulsion to allow r to grow?
                    # If we keep them far, r can be larger.
                    # But we are limited by box size.
                    # So maybe only repel if tight?
                    
                    tightness = r_sum - dist
                    if tightness > 0:
                        # Overlap
                        force_mag = tightness * 10.0
                        fx += (dx / dist) * force_mag
                        fy += (dy / dist) * force_mag
                    else:
                        # If loose, maybe attract slightly to pack them?
                        # No, packing tight is good for area, but we want sum of radii.
                        # Wait, if they are far apart, radii can be larger.
                        # But they can't be too far because of box.
                        # So maybe we don't need attraction.
                        # Just repulsion to fix overlaps is enough?
                        # But we need to find the configuration that allows max radii.
                        # This is like finding a local maximum of the "inradius" function.
                        pass

                forces[i, 0] = fx
                forces[i, 1] = fy
            
            # Update centers
            # Learning rate
            lr = 0.02 * (1.0 - step/1000)
            centers += forces * lr
            
            # Clip to box (loosely, radii check handles boundary)
            centers = np.clip(centers, 0.001, 0.999)
            
            # Check sum
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # Final validation and fixing
    # Ensure valid packing
    # Re-run constraint satisfaction one last time
    centers = best_centers
    radii = best_radii
    
    # Fix overlaps by shrinking radii (last resort)
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-12:
                    sum_r = radii[i] + radii[j]
                    scale = dist / sum_r
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            min_dist = min(x, 1-x, y, 1-y)
            if r > min_dist:
                radii[i] = min_dist
                changed = True
        if not changed:
            break
            
    return centers, radii, np.sum(radii)
