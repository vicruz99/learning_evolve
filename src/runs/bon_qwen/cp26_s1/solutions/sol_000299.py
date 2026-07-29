# sol_000299 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2823a898) state=3816fed2 sum of radii=1.645875 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization
    # Start with a perturbed 5x5 grid for 25 circles, plus one extra
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.01) # Start small
    
    # Create a 5x5 grid
    grid_size = 5
    spacing = 0.2
    offset = 0.1
    
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if idx < n:
                # Add small random perturbation
                centers[idx] = [offset + c * spacing + np.random.uniform(-0.02, 0.02),
                                offset + r * spacing + np.random.uniform(-0.02, 0.02)]
                # Ensure inside bounds initially
                centers[idx][0] = np.clip(centers[idx][0], 0.1, 0.9)
                centers[idx][1] = np.clip(centers[idx][1], 0.1, 0.9)
                idx += 1
    
    # Place the 26th circle in a random valid spot if not filled (though loop fills 25)
    # The loop above fills 25. We need 26.
    if n > 25:
        # Place last one near center or random gap
        centers[25] = [0.5 + np.random.uniform(-0.1, 0.1), 
                       0.5 + np.random.uniform(-0.1, 0.1)]
        centers[25] = np.clip(centers[25], 0.1, 0.9)

    # 2. Simulation Parameters
    # We will expand radii and push circles apart
    # Using a simple Velocity Verlet / Euler integration with damping
    
    velocities = np.random.randn(n, 2) * 0.01
    dt = 0.05
    damping = 0.9
    repulsion_k = 50.0 # Strength of repulsion
    wall_k = 100.0 # Strength of wall repulsion
    jitter_scale = 0.02 # Initial temperature
    growth_rate = 0.0001 # How fast radii grow per step
    
    max_radii_growth = 0.2 # Cap on radius
    
    # Number of iterations
    num_steps = 8000
    
    # Current target radii for growth logic
    current_r = 0.01
    radii[:] = current_r

    for step in range(num_steps):
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Vectorized distance calculation is better but O(N^2) is fine for N=26
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap
                    overlap = min_dist - dist
                    # Force direction: push i away from j
                    dir_vec = diff / dist
                    # Force magnitude proportional to overlap
                    f_mag = repulsion_k * overlap
                    forces[i] += f_mag * dir_vec
                    forces[j] -= f_mag * dir_vec
                elif dist < 1e-9:
                    # Coincident, push random
                    forces[i] += np.random.randn(2)
                    forces[j] -= np.random.randn(2)

        # Wall repulsion
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += wall_k * (r - x)
            # Right wall
            elif x > 1 - r:
                forces[i, 0] -= wall_k * (x - (1 - r))
            
            # Bottom wall
            if y < r:
                forces[i, 1] += wall_k * (r - y)
            # Top wall
            elif y > 1 - r:
                forces[i, 1] -= wall_k * (y - (1 - r))
        
        # Apply jitter (Simulated Annealing)
        # Temperature decays over time
        temperature = jitter_scale * math.exp(-step / 2000)
        forces += np.random.randn(n, 2) * temperature
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        centers += velocities * dt
        
        # Clamp positions to valid range [r, 1-r] to prevent escaping
        # Note: We clamp to [0, 1] effectively, but radii constraint handled by force
        # Better to hard clamp to [0, 1] to be safe, force handles the 'r' buffer
        centers[:, 0] = np.clip(centers[:, 0], 0.0, 1.0)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, 1.0)
        
        # Grow radii slowly
        # Only grow if step is advanced enough to allow settling
        if step > 100:
            growth = growth_rate
            # Maybe grow faster initially?
            if step < 1000:
                growth = growth_rate * 2
            
            new_r = radii[0] + growth # Assume uniform for simplicity of growth logic?
            # Actually, let's just increase all radii uniformly to maximize sum
            # But we must ensure they don't blow up.
            # A better strategy: increase radii based on available space?
            # Uniform growth is simple and works well with repulsion.
            
            # Check if uniform growth is safe or if we should vary?
            # Varying radii is harder to control. Let's stick to uniform growth 
            # but allow the solver to find positions for equal radii first,
            # then we can relax?
            # Actually, the problem asks to maximize sum of radii.
            # Equal radii is a good baseline.
            
            # Let's increase radii uniformly
            radii += growth
            
            # Cap radius
            if radii[0] > 0.2: # 0.2 is half width, impossible for 26 circles
                 radii[:] = 0.19 # Hard cap, though optimal is around 0.1

    # 3. Final Adjustment
    # After simulation, radii are roughly equal. 
    # Now, compute the exact max radius for each circle given the final centers.
    # This ensures no overlap and validity.
    
    final_radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        r_boundary = min(centers[i, 0], 1 - centers[i, 0], 
                         centers[i, 1], 1 - centers[i, 1])
        
        # Distance to other circles
        r_neighbors = np.inf
        for j in range(n):
            if i == j: continue
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            # Max radius for i is half the distance to j (assuming j has radius 0? No)
            # We need to solve for radii simultaneously?
            # Simple greedy approach: 
            # r_i = min( boundary_dist, min_j (dist_ij / 2) )?
            # No, that assumes r_j = r_i.
            
            # Better: 
            # r_i <= dist_ij - r_j.
            # This is coupled.
            # However, since we just pushed them apart, they are likely touching.
            # If we fix centers, the optimal radii are determined by the "tightest" constraints.
            # A simple valid assignment is r_i = min(boundary_dist, min_j(dist_ij / 2)).
            # Let's verify: if r_i = d_ij/2 and r_j = d_ij/2, then r_i + r_j = d_ij. Valid.
            # Is it possible to have larger r_i?
            # Only if r_j is smaller. But we want to maximize sum.
            # If we reduce r_j to increase r_i, sum might not change or decrease.
            # Equal radii is locally optimal for sum given fixed centers?
            # Yes, for fixed centers, sum is max when radii are equal (if constrained by same neighbors).
            # Actually, r_i is constrained by different neighbors.
            # But usually, setting r_i = min(...) works well.
            
            min_dist_to_other = np.inf
            for j in range(n):
                if i == j: continue
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < min_dist_to_other:
                    min_dist_to_other = d
            
            # The radius is limited by the closest neighbor.
            # If we set r_i = min_dist_to_other / 2, we ensure r_i + r_j <= d_ij is NOT guaranteed 
            # because r_j might also be limited by something else to be large.
            # Example: 3 circles in line. 1-2 dist 10, 2-3 dist 10.
            # r_2 limited by 1 and 3. r_2 <= 5.
            # r_1 limited by 2. r_1 <= 10 - r_2 = 5.
            # So r_i = 5 works.
            # If 1-2 dist 2, 2-3 dist 100.
            # r_2 <= 1 (from 1). r_1 <= 1 (from 2). r_3 <= 50 (from 2).
            # Sum = 52.
            # If we set r_i = dist/2:
            # r_1 = 1, r_2 = 1 (limited by 1), r_3 = 50.
            # Check: r_1+r_2 = 2 = dist(1,2). OK.
            # r_2+r_3 = 51 <= 100. OK.
            
            # So r_i = min( boundary_dist, min_{j!=i} (dist_ij / 2) ) is a VALID assignment.
            # Does it maximize sum?
            # In the example above: r_2 is limited by r_1?
            # Actually r_2 is limited by dist(1,2)/2 = 1.
            # r_1 is limited by dist(1,2)/2 = 1.
            # This works.
            
            # Is it possible to increase sum?
            # If r_2 is limited by dist(1,2)/2, then r_2 + r_1 <= dist(1,2) implies r_1 <= dist - r_2.
            # If we pick r_2 = dist/2, then r_1 <= dist/2.
            # So we can't increase r_1 beyond dist/2 without decreasing r_2.
            # Sum r_1 + r_2 <= dist.
            # So capping at dist/2 is optimal for the pair sum?
            # Yes, r_1 + r_2 <= d_12. Max sum is d_12 when r_1=r_2=d_12/2.
            # So for any pair, the sum of radii is bounded by distance.
            # Thus, setting each radius to min(dist/2) is a safe greedy strategy that often yields high sum.
            # It might not be globally optimal if a circle is constrained by multiple neighbors,
            # but it's a very good heuristic.
            
            r_candidate = min_dist_to_other / 2.0
            final_radii[i] = min(r_boundary, r_candidate)

    # Recalculate sum
    sum_radii = np.sum(final_radii)
    
    # Validate internally (just to be sure)
    # If any overlap, reduce radii slightly? 
    # The formula r_i <= d_ij/2 ensures r_i + r_j <= d_ij.
    # So it should be valid.
    
    return centers, final_radii, float(sum_radii)
