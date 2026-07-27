# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bf51a1cd) state=06baeb89 sum of radii=1.405274 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed growing circles algorithm.
    """
    n_circles = 26
    np.random.seed(42) # For reproducibility

    # 1. Initialization
    # We initialize centers in a staggered grid pattern to mimic hexagonal packing.
    # We aim for roughly 5-6 circles per row.
    # Let's try to distribute them in 6 rows.
    # 26 circles: maybe 5 rows of 5 and 1 extra? Or 6 rows.
    # Let's generate a set of points that are well-distributed.
    centers = np.zeros((n_circles, 2))
    
    # Using a simple heuristic to place points:
    # We can try to fill a grid and pick the best spots, or just use a low-discrepancy sequence.
    # Or simply a hexagonal grid subset.
    
    # Let's try to place them in rows. 
    # If we have 6 rows, average 4.33 circles per row.
    # If we have 5 rows, average 5.2.
    # 5 rows of 5 is 25. 1 extra.
    # Let's place 25 in a 5x5 grid first, then perturb.
    
    # Actually, a random uniform initialization followed by strong repulsion often works well 
    # if the simulation is long enough, but structured init is faster.
    # Let's use a Poisson-disk-like sampling or just a dense grid.
    
    # Generate a grid of points
    # 6x6 grid = 36 points. We need 26.
    # We can pick points to maximize min-distance.
    # But for simplicity, let's just pick 26 points from a 6x6 grid.
    
    grid_x = np.linspace(1/12, 11/12, 6) # Spacing 1/6
    grid_y = np.linspace(1/12, 11/12, 6)
    
    points = []
    for x in grid_x:
        for y in grid_y:
            points.append([x, y])
            if len(points) == n_circles:
                break
        if len(points) == n_circles:
            break
            
    centers = np.array(points)
    
    # Add some small random jitter to break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    
    # Clip to valid range [0.05, 0.95] initially
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n_circles, 0.005) # Start small
    
    # 2. Optimization Parameters
    iterations = 20000
    growth_rate_init = 5e-5
    repulsion_strength = 10.0
    wall_strength = 20.0
    damping = 0.5 # Velocity damping
    
    velocities = np.zeros((n_circles, 2))
    
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    # 3. Main Loop
    for step in range(iterations):
        # Annealing parameters
        progress = step / iterations
        growth_rate = growth_rate_init * (1.0 - progress * 0.8) # Slow down growth
        repulsion_strength = 10.0 * (1.0 + progress * 2.0) # Increase repulsion to tighten
        
        # --- Growth Phase ---
        # Try to grow radii slightly
        # We grow all radii by a small amount
        current_growth = growth_rate
        radii += current_growth
        
        # --- Force Calculation ---
        forces = np.zeros((n_circles, 2))
        
        for i in range(n_circles):
            xi, yi = centers[i]
            ri = radii[i]
            
            fx, fy = 0.0, 0.0
            
            # Wall Repulsion
            # Left wall
            if xi - ri < 0:
                overlap = -(xi - ri)
                fx += wall_strength * overlap
            # Right wall
            if xi + ri > 1:
                overlap = xi + ri - 1
                fx -= wall_strength * overlap
            # Bottom wall
            if yi - ri < 0:
                overlap = -(yi - ri)
                fy += wall_strength * overlap
            # Top wall
            if yi + ri > 1:
                overlap = yi + ri - 1
                fy -= wall_strength * overlap
            
            # Inter-circle Repulsion
            for j in range(i + 1, n_circles):
                xj, yj = centers[j]
                rj = radii[j]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                
                min_dist = ri + rj
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap amount
                    overlap = min_dist - dist
                    # Force magnitude proportional to overlap
                    # Force vector direction is from j to i
                    force_mag = repulsion_strength * overlap
                    fx += force_mag * (dx / dist)
                    fy += force_mag * (dy / dist)
                    
                    # Apply equal and opposite force to j
                    forces[j, 0] -= force_mag * (dx / dist)
                    forces[j, 1] -= force_mag * (dy / dist)
            
            forces[i, 0] = fx
            forces[i, 1] = fy
            
        # --- Update Positions ---
        # Update velocities with forces
        velocities += forces * 0.001 # Small time step for stability
        velocities *= damping # Damping
        
        centers += velocities
        
        # Hard constraints: clamp centers to keep them inside valid radius range
        # This helps prevent flying off if forces are huge
        # But we must be careful not to trap them in walls if radii are large.
        # Ideally, forces handle this.
        # However, numerical instability can push centers out.
        # Let's enforce center bounds loosely: [0, 1]
        # But strictly, center must be in [r, 1-r].
        # Let's just clamp to [0, 1] and rely on wall forces to push back.
        centers = np.clip(centers, 0.0, 1.0)
        
        # Check validity and track best
        # We only count valid packings (no overlaps) as "good" to track best
        # But during optimization, overlaps are allowed and resolved by forces.
        # We can check validity occasionally.
        if step % 1000 == 0:
            is_valid = True
            # Check boundaries
            for i in range(n_circles):
                x, y = centers[i]
                r = radii[i]
                if x < r or x > 1-r or y < r or y > 1-r:
                    is_valid = False
                    break
            if is_valid:
                # Check overlaps
                for i in range(n_circles):
                    for j in range(i+1, n_circles):
                        dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                        if dist < radii[i] + radii[j] - 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    s = np.sum(radii)
                    if s > best_sum_radii:
                        best_sum_radii = s
                        best_centers = centers.copy()
                        best_radii = radii.copy()

    # 4. Final Cleanup and Validation
    # Return the best valid state found, or the current state if we want to risk it.
    # The problem requires a valid packing.
    # If our best tracked state is valid, use it.
    # Otherwise, we might need to shrink radii slightly to ensure validity.
    
    if best_centers is not None:
        final_centers = best_centers
        final_radii = best_radii
    else:
        final_centers = centers
        final_radii = radii
        
    # Post-processing: ensure strict validity
    # Sometimes numerical errors leave tiny overlaps.
    # We can shrink radii slightly if needed, but that hurts score.
    # Let's verify and if invalid, shrink radii minimally.
    
    # Check overlaps
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dist = np.sqrt((final_centers[i,0]-final_centers[j,0])**2 + (final_centers[i,1]-final_centers[j,1])**2)
            sum_r = final_radii[i] + final_radii[j]
            if dist < sum_r - 1e-12:
                # Overlap detected, shrink both radii proportionally to resolve
                # Or just shrink one.
                # Simple fix: reduce radii until valid.
                # This is a hack, but ensures valid output.
                # Better: run a few more steps with 0 growth and high repulsion.
                pass
                
    # Let's run a final "cooldown" phase to resolve any minor overlaps without growing
    for _ in range(1000):
        forces = np.zeros((n_circles, 2))
        for i in range(n_circles):
            xi, yi = final_centers[i]
            ri = final_radii[i]
            fx, fy = 0.0, 0.0
            
            # Wall repulsion
            if xi - ri < 0: fx += wall_strength * (-(xi - ri))
            if xi + ri > 1: fx -= wall_strength * (xi + ri - 1)
            if yi - ri < 0: fy += wall_strength * (-(yi - ri))
            if yi + ri > 1: fy -= wall_strength * (yi + ri - 1)
            
            for j in range(i + 1, n_circles):
                xj, yj = final_centers[j]
                rj = final_radii[j]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    force_mag = repulsion_strength * overlap * 2 # Stronger
                    fx += force_mag * (dx / dist)
                    fy += force_mag * (dy / dist)
                    forces[j, 0] -= force_mag * (dx / dist)
                    forces[j, 1] -= force_mag * (dy / dist)
            forces[i, 0] = fx
            forces[i, 1] = fy
            
        final_centers += forces * 0.0005 # Very small step
        
    # Final check and clipping
    for i in range(n_circles):
        r = final_radii[i]
        # Ensure center is valid
        final_centers[i, 0] = max(r, min(1-r, final_centers[i, 0]))
        final_centers[i, 1] = max(r, min(1-r, final_centers[i, 1]))
        
        # If radius is too big for square (impossible), shrink it
        if 2*r > 1:
            final_radii[i] = 0.5
            
    final_sum = np.sum(final_radii)
    
    return final_centers, final_radii, float(final_sum)

# To verify locally
if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    
    # Manual validation check
    n = centers.shape[0]
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} out of bounds")
            valid = False
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap")
                valid = False
    print(f"Valid: {valid}")
