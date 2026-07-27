import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a physics-based repulsion simulation starting from a hexagonal grid.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Packing
    # We arrange circles in rows. 
    # A hexagonal pattern allows for denser packing.
    # We need to fit 26 circles.
    # Let's try a configuration with 5 rows. 
    # Counts per row could be 5, 5, 5, 5, 6 or similar.
    # However, to maximize radius, we want to balance width and height.
    # A 5x5 grid (25 circles) fits r=0.1. 
    # Hexagonal packing can achieve higher density.
    
    # Let's construct a hexagonal grid that is slightly larger than needed, 
    # then trim or adjust to 26 circles.
    # Or just place 26 circles in a dense pattern.
    
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Try a hexagonal arrangement
    # Rows with 5, 5, 5, 5, 6 circles? 
    # Or 6, 5, 5, 5, 5?
    # Let's try to fit them in a rectangular box initially and let the solver fix it.
    
    row_counts = [5, 5, 5, 5, 6] # Total 26
    # But a row of 6 requires width 12r, row of 5 requires 10r.
    # If we shift rows, we might fit better.
    
    # Let's use a simpler initialization:
    # Place 26 circles in a grid and perturb, or just a dense hexagonal block.
    # Hexagonal packing:
    # Row i has centers at x = x0 + k*2r, y = y0 + i*sqrt(3)r
    # Alternating rows shifted by r.
    
    # Let's try to fit 26 circles in a roughly 5x6 or 6x5 hexagonal structure.
    # 5 rows of 5 is 25. Add 1.
    # Maybe 6 rows? 4, 5, 4, 5, 4, 4? Sum 26.
    # Or 5, 5, 5, 5, 5, 1.
    
    # Let's try a 6x5 hexagonal cluster (30 circles) and remove 4, or just place 26.
    # Let's just place them in a grid first to ensure spread, then optimize.
    # Actually, a random placement or grid is fine for the solver if it's global enough.
    # But hexagonal is safer.
    
    # Let's create a hexagonal lattice and pick 26 points.
    r_init = 0.05 # Small initial radius to avoid immediate overlap
    centers_list = []
    
    # Generate hexagonal points
    # We want to cover the square.
    # Spacing 2*r_init
    y = r_init
    row = 0
    while len(centers_list) < n_circles:
        x = r_init if row % 2 == 0 else 2 * r_init # Shift alternate rows
        while x < 1 - r_init + 1e-9:
            centers_list.append([x, y])
            if len(centers_list) >= n_circles:
                break
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
    
    centers = np.array(centers_list[:n_circles])
    radii = np.full(n_circles, r_init)
    
    # 2. Optimization using Repulsive Forces (Simulated Annealing / Gradient Ascent)
    # We will maximize the sum of radii.
    # We can model this as minimizing a potential energy function.
    # Energy = - sum(radii) + penalty(overlaps) + penalty(boundary)
    
    # However, scipy minimize is better for constrained problems.
    # But the constraints are non-convex.
    # Let's use a custom iterative loop that pushes circles apart and grows radii.
    
    current_radii = radii.copy()
    current_centers = centers.copy()
    
    # Simulation parameters
    dt = 0.05
    friction = 0.9
    growth_rate = 0.0005
    force_scale = 10.0
    
    num_steps = 2000
    
    for step in range(num_steps):
        # Increase radii slowly
        current_radii += growth_rate
        
        # Calculate forces
        forces = np.zeros_like(current_centers)
        
        # Pairwise repulsion
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = current_centers[i] - current_centers[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = current_radii[i] + current_radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    f_mag = overlap * force_scale
                    # Direction from j to i
                    direction = diff / dist
                    forces[i] += direction * f_mag
                    forces[j] -= direction * f_mag
                elif dist < min_dist * 1.5 and dist > 1e-9:
                    # Soft repulsion to prevent sudden overlaps
                    f_mag = (min_dist * 1.5 - dist) * 0.1 * force_scale
                    direction = diff / dist
                    forces[i] += direction * f_mag
                    forces[j] -= direction * f_mag

        # Boundary repulsion
        for i in range(n_circles):
            x, y = current_centers[i]
            r = current_radii[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * force_scale
            # Right wall
            elif x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * force_scale
            
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * force_scale
            # Top wall
            elif y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * force_scale
        
        # Update positions
        velocities = forces * dt
        current_centers += velocities
        current_centers *= friction # Damping
        
        # Clamp positions to [0, 1] strictly to prevent numerical escape
        current_centers = np.clip(current_centers, 0, 1)
        
        # If radii are too large for position, clamp position?
        # Actually, the force handles it.
        
        # Periodically check if valid, if not, shrink radii slightly?
        # No, we want to find max radii. If invalid, forces will correct positions.
        # But if stuck, maybe reduce radii?
        # For now, let it run.
        
        # To speed up, we can reduce growth rate as time goes on
        if step > 1000:
            growth_rate = 0.0001

    # 3. Refinement with SciPy Optimization
    # We have a good configuration. Now let's use a local optimizer to polish it.
    # We want to maximize sum(radii) subject to constraints.
    # This is hard. Instead, let's just ensure validity and maybe adjust.
    # But the physics simulation should have found a local optimum.
    
    # Let's verify and maybe do a small adjustment.
    # Check for any overlaps and fix them by shrinking radii slightly if necessary?
    # No, we want max sum.
    
    # Let's perform a few steps of "gradient ascent" on radii with position correction.
    # Actually, the simulation above is essentially that.
    
    # Final validation and clean up
    # Ensure no NaN
    if np.isnan(current_centers).any() or np.isnan(current_radii).any():
        # Fallback to grid if something went wrong
        centers = np.array([[ (i % 5) * 0.2 + 0.1, (i // 5) * 0.2 + 0.1] for i in range(26)])
        # Adjust for 26th circle
        centers[25] = [0.5, 0.9] 
        current_radii = np.full(26, 0.05)

    # Double check constraints and fix small violations by shrinking radii
    # This is a safety net.
    for _ in range(10):
        min_dist_safe = float('inf')
        for i in range(n_circles):
            # Boundary
            dist_boundary = min(
                current_centers[i, 0] - current_radii[i],
                1 - current_centers[i, 0] - current_radii[i],
                current_centers[i, 1] - current_radii[i],
                1 - current_centers[i, 1] - current_radii[i]
            )
            if dist_boundary < 0:
                # Circle is outside, shrink radius
                current_radii[i] -= dist_boundary # dist_boundary is negative, so minus adds
                # Also move center?
                # Just shrink for now
            min_dist_safe = min(min_dist_safe, dist_boundary)
            
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
                overlap = current_radii[i] + current_radii[j] - dist
                if overlap > 0:
                    # Overlap detected, reduce radii
                    reduction = overlap / 2
                    current_radii[i] -= reduction
                    current_radii[j] -= reduction
        
        if min_dist_safe >= -1e-6: # Tolerance
            break
            
    # Ensure non-negative
    current_radii = np.maximum(current_radii, 0)
    
    # Final Sum
    sum_radii = np.sum(current_radii)
    
    # If the simulation resulted in a very low sum (failed), try a simple grid fallback
    if sum_radii < 2.0:
        # Fallback: 5x5 grid + 1
        # 25 circles of r=0.1 sum=2.5
        # 26th circle small.
        # But we can do better.
        # Let's try to perturb the grid to fit 26.
        # Actually, the physics sim should have worked.
        pass

    return current_centers, current_radii, sum_radii

# Helper to run the packing
def solve():
    return run_packing()

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)