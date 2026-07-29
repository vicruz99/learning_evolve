# sol_000185 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1380b4f2) state=b4a0e54d sum of radii=1.524627 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def solve_packing_heuristic():
    """
    Uses a combination of simulated annealing and geometric relaxation to pack 26 circles.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Generate initial staggered grid (hexagonal-like)
    # Target roughly 5x5 arrangement
    rows = 5
    cols = 5 # 5*5 = 25, plus 1 extra
    # We will distribute 26 circles. 
    # Let's try 6 rows to utilize vertical space better for staggered packing
    # Row pattern: 5, 5, 5, 5, 5, 1 -> 26? No, 6*5=30.
    # Pattern: 5, 5, 5, 5, 5, 1 is 26.
    # But 5 rows of 5 is 25. Adding 1 might be hard.
    # Let's try 5 rows with counts [5, 5, 5, 6, 5] = 26?
    # 5,5,5,6,5 is 26.
    
    # Better: 5 rows, counts [6, 5, 5, 5, 5] = 26
    counts = [6, 5, 5, 5, 5]
    total_circles = sum(counts)
    assert total_circles == n
    
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.1 # Initial guess
    
    current_idx = 0
    r_est = 0.12 # Initial radius estimate (will be adjusted)
    
    # Vertical spacing for hex packing
    dy = r_est * np.sqrt(3)
    
    # Estimate width required for max column count
    # For count 6: width approx 2*r + (6-1)*2*r = 12r. 
    # 12*0.12 = 1.44 > 1. So we need to scale r down or adjust layout.
    # Let's just place them and let the optimizer fix radii.
    
    # Let's place centers roughly evenly
    # We will map counts to y-coordinates
    y_coords = np.linspace(0.15, 0.85, len(counts))
    
    for row_idx, count in enumerate(counts):
        y = y_coords[row_idx]
        # Shift x for staggered rows
        if row_idx % 2 == 1:
            offset = r_est # Shift by approx r
        else:
            offset = 0
        
        # Distribute count circles in x
        # If count is 6, we need to fit them. 
        # Let's just space them evenly in [0, 1]
        if count > 0:
            # Use linspace with padding
            # Padding roughly r_est
            xs = np.linspace(0.08, 0.92, count)
            
            for k in range(count):
                x = xs[k] + offset * 0.5 # Small offset for staggering
                # Keep within bounds roughly
                x = np.clip(x, 0.05, 0.95)
                centers[current_idx] = [x, y]
                current_idx += 1

    # 2. Simulated Annealing / Relaxation
    # We want to maximize sum of radii. 
    # We will iterate: increase radii, push apart, cool down.
    
    best_sum = 0.0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    current_sum = np.sum(radii)
    
    temp = 1.0
    cooling_rate = 0.995
    
    # Relaxation parameters
    repulsion_strength = 1.0
    radius_growth_rate = 1.0001
    
    for step in range(2000):
        # Grow radii
        radii *= radius_growth_rate
        
        # Repulsion forces (Push apart)
        # Simple gradient ascent on min distance
        forces = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                target_dist = radii[i] + radii[j]
                
                if dist < target_dist + 1e-6:
                    # Overlap
                    # Force proportional to overlap
                    overlap = target_dist - dist
                    if dist > 1e-9:
                        direction = diff / dist
                        force_mag = repulsion_strength * overlap
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
                    else:
                        # Same center, push random
                        forces[i] += np.random.uniform(-0.1, 0.1, 2)
                        forces[j] -= forces[i]
            
            # Boundary forces
            r = radii[i]
            x, y = centers[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += repulsion_strength * (r - x)
            # Right wall
            if x > 1 - r:
                forces[i, 0] -= repulsion_strength * (x - (1 - r))
            # Bottom wall
            if y < r:
                forces[i, 1] += repulsion_strength * (r - y)
            # Top wall
            if y > 1 - r:
                forces[i, 1] -= repulsion_strength * (y - (1 - r))
        
        # Apply forces (velocity Verlet style or simple Euler)
        # Add some noise for SA
        noise = np.random.normal(0, temp * 0.01, centers.shape)
        centers += forces * 0.1 + noise
        
        # Clamp centers to [0,1] roughly, but let forces handle it?
        # Hard clamp to prevent divergence
        centers = np.clip(centers, 1e-5, 1 - 1e-5)
        
        # Accept/Reject step for SA?
        # For simplicity, we just iterate. 
        # If sum decreases too much, maybe reset?
        # But we are growing radii, so we might violate constraints.
        # The forces should resolve violations.
        
        # Check validity periodically
        if step % 100 == 0:
            # If we have a valid configuration, record it
            # We need to shrink radii slightly to ensure validity if we are close
            # But the loop grows them.
            # Let's just track the best valid state found during a "stable" period.
            pass
            
        # Cooling
        temp *= cooling_rate
        
    # Final polishing: Optimize radii for the final configuration
    # We have positions. Maximize radii such that no overlap.
    # This is a local optimization.
    # We can use scipy to maximize sum(r) given fixed centers? No, centers move.
    
    # Let's use a dedicated optimizer for the final step
    # Variables: x_i, y_i, r_i
    # But 78 vars is a lot.
    # Let's fix the topology and just grow radii until tight.
    
    # Simple "inflate" step
    radii[:] = 0.01 # Reset radii to small
    centers = best_centers # Use best positions if available, or current
    
    # Actually, the simulation above grows radii. 
    # Let's run a dedicated optimization function using scipy's minimize with penalty.
    
    return centers, radii

def run_packing():
    """
    Main function to run the packing optimization.
    """
    n = 26
    
    # Strategy:
    # 1. Use Differential Evolution to find a good global configuration.
    #    Since the landscape is rugged, DE is suitable.
    #    We maximize sum of radii.
    
    # To make DE efficient, we bound radii to reasonable range.
    # Max possible radius is 0.5.
    # Min 0.
    
    # Vector: [x1, y1, r1, x2, y2, r2, ...]
    # Length 78.
    
    bounds = [(0, 1)] * (n * 2) + [(0, 0.5)] * n
    # Actually, x,y bounds [0,1]. But r bounds [0, 0.5].
    # However, x and r are coupled.
    # Let's just use bounds [0, 1] for x, y and [0, 0.5] for r.
    # The penalty function will handle the rest.
    
    def fitness(params):
        # params is 1D array of length 78
        # Reshape to centers (26, 2) and radii (26,)
        centers = params[:2*n].reshape((n, 2))
        radii = params[2*n:]
        
        score = 0
        penalty = 0
        
        # Sum of radii
        score = np.sum(radii)
        
        # Penalty for boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # x - r >= 0 => r - x <= 0
            if r - x > 0: penalty += 1000 * (r - x)**2
            if x + r - 1 > 0: penalty += 1000 * (x + r - 1)**2
            if r - y > 0: penalty += 1000 * (r - y)**2
            if y + r - 1 > 0: penalty += 1000 * (y + r - 1)**2
            
            # Ensure radii non-negative (bounds handle this, but just in case)
            if r < 0: penalty += 1000 * r**2
            
        # Penalty for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Quadratic penalty
                    penalty += 1000 * (min_dist - dist)**2
                    
        # We want to maximize score, so minimize negative score + penalty
        return -(score - penalty)

    # DE is slow for 78 dims. Let's try a reduced approach or better initialization.
    # Let's use a heuristic initialization first, then refine with a local optimizer?
    # Or just run DE with low pop size and iterations.
    
    # Let's try a different approach:
    # Fix radii to be roughly equal, optimize positions.
    # Then optimize radii.
    
    # Heuristic Solver
    # 1. Place circles in a grid.
    # 2. Run a custom optimizer.
    
    centers = np.random.rand(n, 2)
    radii = np.full(n, 0.1)
    
    # Iterative improvement
    # We want to maximize sum(r).
    # Let's try to find a valid configuration with high sum.
    
    # Use scipy.optimize.minimize with trust-constr?
    # Constraints are hard.
    
    # Let's stick to the "grow and push" heuristic which is robust.
    
    # Initialize in a grid
    # 5x5 grid + 1
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [xs[i], ys[j]]
            idx += 1
    # 26th circle
    centers[25] = [0.5, 0.5] # Center, will overlap, optimizer will move it
    
    # Reset radii to small to allow movement
    radii[:] = 0.05
    
    # Optimization loop
    # We use a simulated annealing style with radius growth
    
    best_sum = 0
    best_state = (centers.copy(), radii.copy())
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    
    # Parameters
    dt = 0.01
    repulsion = 5.0
    temp = 0.1
    growth = 1.0001
    
    # We run this for a fixed number of steps
    # To make it deterministic and fast enough
    
    for step in range(5000):
        # Grow radii
        current_radii *= growth
        
        # Calculate forces
        forces = np.zeros_like(current_centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = current_centers[i] - current_centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-5:
                    dist = 1e-5
                    diff = np.random.rand(2) # Random push if same center
                
                target = current_radii[i] + current_radii[j]
                
                if dist < target:
                    # Repel
                    force_mag = repulsion * (target - dist) / dist
                    forces[i] += diff * force_mag
                    forces[j] -= diff * force_mag
        
        # Boundary repulsion
        for i in range(n):
            r = current_radii[i]
            x, y = current_centers[i]
            
            # Left
            if x < r:
                forces[i, 0] += repulsion * (r - x)
            # Right
            if x > 1 - r:
                forces[i, 0] -= repulsion * (x - (1 - r))
            # Bottom
            if y < r:
                forces[i, 1] += repulsion * (r - y)
            # Top
            if y > 1 - r:
                forces[i, 1] -= repulsion * (y - (1 - r))
                
        # Update centers
        # Add noise
        noise = np.random.normal(0, temp, current_centers.shape)
        current_centers += forces * dt + noise
        
        # Clamp
        current_centers = np.clip(current_centers, 1e-6, 1 - 1e-6)
        
        # Cooling
        if step > 1000:
            temp *= 0.999
            growth = 1.00005 # Slow down growth
        else:
            growth = 1.0001
            
        # Check validity and record best
        # We need to check if current configuration is valid
        # But forces might be pushing them apart, so radii might be too big for current positions.
        # We want to record the state where radii were largest valid.
        
        # Actually, the algorithm grows radii, so validity is lost.
        # We should validate with a slight shrink or just check raw.
        # If invalid, the forces will correct it.
        # The "best" sum is the sum of radii when the system stabilizes.
        
        # Let's check validity at the end? 
        # No, we need to track max valid sum.
        
        # Heuristic: if no overlaps and inside, record.
        # But with forces, overlaps are common.
        
        # Alternative: After simulation, take the centers, and compute max radii?
        # No, radii are coupled.
        
        # Let's just run the simulation and return the final state, 
        # then shrink radii slightly to ensure validity.
        
    # Final adjustment:
    # The simulation might end with overlaps.
    # We need to reduce radii until valid.
    
    # Check validity
    valid = False
    for k in range(50):
        valid = validate_packing(current_centers, current_radii)
        if valid:
            break
        # Shrink radii
        current_radii *= 0.99
        
    # Calculate sum
    sum_radii = np.sum(current_radii)
    
    # Try to improve by local optimization using scipy?
    # Maybe too risky if time limit.
    
    # Let's try one more pass: 
    # Use the final centers as anchor, maximize radii?
    # No, centers should move too.
    
    # The heuristic above is a form of optimization.
    # With 5000 steps and growth, it should find a decent packing.
    
    # Let's ensure radii are positive
    current_radii = np.maximum(current_radii, 1e-6)
    
    # Re-validate strictly
    if not validate_packing(current_centers, current_radii):
        # Fallback to a known valid packing (e.g. grid with smaller radii)
        # Grid 5x5, r=0.1 -> sum 2.5. Plus 1 small?
        # Let's construct a valid grid packing
        centers_safe = np.zeros((n, 2))
        radii_safe = np.zeros(n)
        idx = 0
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                centers_safe[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                radii_safe[idx] = 0.1
                idx += 1
        # 26th circle in a gap? 
        # Gap at (0.2, 0.2) relative to (0.1, 0.1)?
        # Distance to (0.1, 0.1) is 0.141. r=0.1. Space 0.041.
        # Place at (0.5, 0.5) is occupied.
        # Let's place at (0.2, 0.4)?
        # Dist to (0.1, 0.3) -> sqrt(0.1^2 + 0.1^2) = 0.141.
        # Dist to (0.3, 0.3) -> 0.141.
        # Dist to (0.1, 0.5) -> 0.141.
        # Dist to (0.3, 0.5) -> 0.141.
        # So radius 0.041 fits.
        centers_safe[25] = [0.2, 0.4]
        radii_safe[25] = 0.04
        sum_safe = np.sum(radii_safe)
        
        # The optimized sum should be better.
        # If our simulation failed badly, use safe.
        if sum_radii < sum_safe:
            current_centers = centers_safe
            current_radii = radii_safe
            sum_radii = sum_safe

    return current_centers, current_radii, sum_radii
