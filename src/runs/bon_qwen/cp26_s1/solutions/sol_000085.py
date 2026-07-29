# sol_000085 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e9cb3956) state=f915a95e sum of radii=0.561913 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses a hexagonal initialization followed by a force-directed optimization.
    """
    n = 26
    # Constants for simulation
    dt = 0.01          # Time step for position updates
    dr = 0.0005        # Radius growth rate
    repulsion_strength = 2.0
    damping = 0.9      # Velocity damping
    shake_magnitude = 0.05 # Random shake to escape local minima
    iterations = 3000  # Number of optimization steps

    # 1. Initialization: Hexagonal Grid
    # We want to place 26 circles. 
    # A hexagonal packing of 26 circles can be arranged in rows.
    # Let's try to fit them in a roughly square-ish block.
    # Approximate number of rows needed: sqrt(26 * sqrt(3) / 2) ~ 6 rows?
    # Let's generate a hexagonal lattice and pick the first 26 points that fit.
    
    # Generate a dense hexagonal grid in a slightly larger square, then fit?
    # Or just generate points and filter.
    # Let's try a specific pattern: 5 rows.
    # Row counts: 5, 5, 6, 5, 5 -> 26? No, 5+5+6+5+5 = 26.
    # But hexagonal rows alternate shift.
    # Let's try: 6, 5, 6, 5, 4 -> 26.
    
    # Let's use a generic hexagonal generator
    centers = []
    
    # Parameters for hex grid generation
    # We want to fill the square [0,1]x[0,1]
    # We will generate points with spacing 1.0 initially, then we will scale/fit later?
    # Actually, let's just place them with some spacing.
    
    # Heuristic: 26 circles. 
    # Let's try to place them in a 5x5 grid + 1, but that's sparse.
    # Let's try to fit a hexagonal packing.
    # Rows at y = 0, h, 2h, ...
    # Points at x = 0, w, 2w... and shifted by w/2 for odd rows.
    
    # Let's just generate a bunch of hexagonal points and select 26 that fit well?
    # Or simpler: Random initialization with high density.
    
    # Better Initialization:
    # Place centers on a hexagonal lattice scaled to fit 26 circles with radius ~0.1
    # Radius 0.1 => diameter 0.2.
    # Spacing 0.2 horizontally, 0.173 vertically.
    
    r_init = 0.05 # Start small
    width = 2 * r_init
    height = math.sqrt(3) * r_init
    
    # Generate points
    points = []
    y = r_init
    row = 0
    while y < 1.0 - r_init:
        x = r_init
        # Shift for odd rows
        shift = width / 2.0 if row % 2 == 1 else 0.0
        x += shift
        
        while x < 1.0 - r_init:
            points.append((x, y))
            x += width
        y += height
        row += 1
        
    # We might have generated more or fewer than 26.
    # If more, select 26 random ones? Or subset?
    # If fewer, fill with random?
    
    # Let's just take the first 26 points.
    # If we have fewer, we need to add random points.
    
    # Actually, with r_init=0.05, spacing is 0.1 and 0.0866.
    # Grid size ~ 10x10 = 100 points. We have plenty.
    
    # Select 26 points. To maximize density, maybe pick points that are well spread?
    # Just picking the first 26 (which fills from bottom-left) is fine.
    # But maybe a random subset is better to avoid clustering?
    # Let's shuffle and pick.
    random.shuffle(points)
    if len(points) < n:
        # Fill remaining with random points in [0.1, 0.9]
        while len(points) < n:
            points.append((random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)))
    
    centers = np.array(points[:n])
    radii = np.full(n, r_init)
    
    # 2. Optimization Loop
    velocities = np.zeros((n, 2))
    
    # We will run the simulation for a fixed number of steps
    # We can dynamically adjust parameters
    
    for step in range(iterations):
        # Decay parameters
        current_dt = dt * (1.0 / (1.0 + step * 0.001))
        current_dr = dr * (1.0 / (1.0 + step * 0.002))
        current_shake = shake_magnitude * (0.999 ** step) # Decay shake
        
        # Grow radii
        radii += current_dr
        
        # Apply shake (random perturbation)
        if random.random() < 0.1: # 10% chance to shake all
            centers += np.random.uniform(-current_shake, current_shake, size=(n, 2))
        
        # Calculate forces
        forces = np.zeros((n, 2))
        
        # Check pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    if dist < 1e-9:
                        # Prevent division by zero, push randomly
                        forces[i] += np.random.uniform(-0.1, 0.1, 2)
                        forces[j] -= np.random.uniform(-0.1, 0.1, 2)
                    else:
                        overlap = min_dist - dist
                        # Repulsive force proportional to overlap
                        # Normalize vector
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Force magnitude
                        f = repulsion_strength * overlap
                        
                        forces[i, 0] -= f * nx
                        forces[i, 1] -= f * ny
                        forces[j, 0] += f * nx
                        forces[j, 1] += f * ny

        # Check boundary constraints and apply forces
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            
            # Left boundary
            if x - r < 0:
                forces[i, 0] += repulsion_strength * (r - x)
            # Right boundary
            if x + r > 1:
                forces[i, 0] -= repulsion_strength * (x + r - 1)
            # Bottom boundary
            if y - r < 0:
                forces[i, 1] += repulsion_strength * (r - y)
            # Top boundary
            if y + r > 1:
                forces[i, 1] -= repulsion_strength * (y + r - 1)

        # Update velocities and positions
        # Add force to velocity
        velocities += forces * current_dt
        # Damping
        velocities *= damping
        # Update positions
        centers += velocities * current_dt
        
        # Hard clamp to keep inside [0, 1] roughly, to prevent explosion
        # Though boundary forces should handle it.
        # Just clamp to [0, 1] for safety
        centers = np.clip(centers, 0, 1)
        
        # Hard clamp radii? 
        # If radii grow too large, constraints will be violated heavily, 
        # but forces will push centers. 
        # However, if a circle is in a corner, it might not fit.
        # We should clamp radii to max possible for current center?
        # Or just let forces handle it. 
        # Let's clamp radii to min(x, 1-x, y, 1-y) to strictly satisfy boundary constraint at every step?
        # No, that prevents optimization. 
        # Better to let them grow and let forces push centers out.
        
        # However, to prevent NaNs or huge numbers, maybe clamp radii max?
        # Max radius is 0.5.
        radii = np.clip(radii, 0, 0.5)

    # 3. Post-processing / Relaxation
    # After the simulation, we might still have slight overlaps due to discretization.
    # We can run a few steps of "resolution only" (no growth) to settle.
    
    for step in range(500):
        forces = np.zeros((n, 2))
        
        # Check pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    if dist < 1e-9:
                         forces[i] += np.random.uniform(-0.05, 0.05, 2)
                         forces[j] -= np.random.uniform(-0.05, 0.05, 2)
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        f = repulsion_strength * overlap # Stronger force to fix
                        forces[i, 0] -= f * nx
                        forces[i, 1] -= f * ny
                        forces[j, 0] += f * nx
                        forces[j, 1] += f * ny

        # Boundary forces
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            if x - r < 0: forces[i, 0] += 10.0 * (r - x)
            if x + r > 1: forces[i, 0] -= 10.0 * (x + r - 1)
            if y - r < 0: forces[i, 1] += 10.0 * (r - y)
            if y + r > 1: forces[i, 1] -= 10.0 * (y + r - 1)

        velocities += forces * 0.005
        velocities *= 0.5
        centers += velocities * 0.005
        centers = np.clip(centers, 0, 1)

    # 4. Final Adjustment
    # If any circles are still slightly out of bounds or overlapping, 
    # we can reduce radii slightly to ensure validity.
    # Or just return as is if valid.
    
    # Check validity and fix if necessary
    valid = True
    # Check boundaries
    for i in range(n):
        r = radii[i]
        x, y = centers[i]
        # Ensure strictly inside with some margin? 
        # The validator allows 1e-12 error.
        # We can just clamp center to valid region if radius is fixed?
        # But radius is variable.
        # If a circle is touching boundary, it's valid.
        # If it's outside, we must move it or shrink radius.
        
        # If outside, shrink radius to fit.
        min_dist_boundary = min(x, 1-x, y, 1-y)
        if min_dist_boundary < r:
            radii[i] = max(0, min_dist_boundary)
            
    # Check overlaps
    # If overlap, shrink the smaller radius? Or both?
    # Shrinking reduces objective, so try to avoid.
    # But we need a valid packing.
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            required_dist = radii[i] + radii[j]
            
            if dist < required_dist:
                # Overlap detected. Reduce radii.
                # Reduce proportionally or just reduce one?
                # Let's reduce both by half the overlap.
                overlap = required_dist - dist
                shrink = overlap / 2.0
                radii[i] -= shrink
                radii[j] -= shrink
                # Ensure non-negative
                radii[i] = max(0, radii[i])
                radii[j] = max(0, radii[j])

    # Final check for negative radii (shouldn't happen but safe)
    radii = np.maximum(radii, 0)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)
