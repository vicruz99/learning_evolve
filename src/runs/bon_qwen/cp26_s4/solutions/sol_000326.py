# sol_000326 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3035e37b) state=30ba8030 sum of radii=1.820000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float sum of radii
    """
    n = 26
    # Initial parameters
    # We start with a radius that is safely small
    current_r = 0.05
    
    # Initialize centers in a grid-like pattern to avoid clumping
    # 6 columns, 5 rows roughly
    cols = 6
    rows = 5
    centers = np.zeros((n, 2))
    idx = 0
    
    # Grid spacing
    step_x = 1.0 / (cols + 1)
    step_y = 1.0 / (rows + 1)
    
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                x = step_x * (c + 1)
                y = step_y * (r + 1)
                centers[idx] = [x, y]
                idx += 1
            else:
                break
        if idx >= n:
            break
            
    radii = np.full(n, current_r)
    
    # Simulation parameters
    # We will run a number of iterations, slowly increasing radius
    # and optimizing positions.
    
    num_iterations = 1000
    # Initial radius increase per step
    r_step = 0.002 
    # Position update coefficient (learning rate)
    alpha = 0.1
    
    # We'll store the best valid state found
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Precompute indices for pairs to speed up loop
    # i < j
    pairs_i = []
    pairs_j = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_i.append(i)
            pairs_j.append(j)
    pairs_i = np.array(pairs_i)
    pairs_j = np.array(pairs_j)
    num_pairs = len(pairs_i)
    
    # Boundary constraints logic
    # We want to keep circles inside [0,1]x[0,1]
    
    for step in range(num_iterations):
        # 1. Increase radius slightly
        current_r += r_step
        radii[:] = current_r
        
        # 2. Compute forces
        forces = np.zeros_like(centers)
        total_overlap = 0.0
        
        # 2a. Pairwise repulsion
        # Vectorized calculation for pairs
        c_i = centers[pairs_i] # Shape (num_pairs, 2)
        c_j = centers[pairs_j]
        
        diff = c_i - c_j
        dists = np.linalg.norm(diff, axis=1) # Shape (num_pairs,)
        
        # Avoid division by zero
        valid_mask = dists > 1e-9
        diff = diff[valid_mask]
        dists = dists[valid_mask]
        
        # Calculate required separation
        # Since radii are equal, r_i + r_j = 2 * current_r
        required_dist = 2 * current_r
        
        overlap = required_dist - dists
        overlap[overlap < 0] = 0 # Only repel if overlapping
        
        total_overlap += np.sum(overlap)
        
        # Force magnitude proportional to overlap
        # Normalize direction
        if np.any(overlap > 0):
            dirs = diff / dists[:, np.newaxis] # Shape (num_valid, 2)
            # The overlap array corresponds to valid_mask
            # We need to map forces back to original indices
            # This is tricky with masking. 
            # Let's use a simpler loop or careful indexing.
            
            # Re-doing without mask for simplicity in logic, though slower
            # N=26 is small, O(N^2) is 325 ops.
            pass

        # Let's revert to explicit loop for clarity and correctness with masking
        forces = np.zeros_like(centers)
        total_energy = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < 2 * current_r:
                    overlap = 2 * current_r - dist
                    total_energy += overlap**2
                    if dist > 1e-9:
                        force_vec = (overlap / dist) * (centers[i] - centers[j])
                        forces[i] += force_vec
                        forces[j] -= force_vec
                    else:
                        # Push randomly if on top of each other
                        forces[i] += np.random.normal(0, 0.01, 2)
                        forces[j] -= forces[i]

        # 2b. Boundary repulsion
        # Walls: x < r, x > 1-r, y < r, y > 1-r
        # Force pushes center away from wall
        
        # Left wall (x < r)
        violation = current_r - centers[:, 0]
        mask = violation > 0
        forces[mask, 0] += violation[mask]
        total_energy += np.sum(violation[mask]**2)
        
        # Right wall (x > 1-r)
        violation = centers[:, 0] - (1.0 - current_r)
        mask = violation > 0
        forces[mask, 0] -= violation[mask]
        total_energy += np.sum(violation[mask]**2)
        
        # Bottom wall (y < r)
        violation = current_r - centers[:, 1]
        mask = violation > 0
        forces[mask, 1] += violation[mask]
        total_energy += np.sum(violation[mask]**2)
        
        # Top wall (y > 1-r)
        violation = centers[:, 1] - (1.0 - current_r)
        mask = violation > 0
        forces[mask, 1] -= violation[mask]
        total_energy += np.sum(violation[mask]**2)
        
        # 3. Update positions
        # Scale forces by alpha
        # Add some damping or just direct update
        # To prevent oscillation, we can limit max displacement
        max_disp = 0.05
        displacement = forces * alpha
        np.clip(displacement, -max_disp, max_disp, out=displacement)
        
        centers += displacement
        
        # 4. Ensure centers stay within [0, 1] strictly to prevent numerical issues
        # Although forces push them in, explicit clipping is safe
        np.clip(centers, 0, 1, out=centers)
        
        # 5. Check if we are stuck or if optimization is converging
        # If energy is low, we are good. If energy is high, we might need to reduce r.
        # However, with simple gradient descent on positions, we might just oscillate.
        
        # Cooling schedule
        if step > 500:
            r_step *= 0.99
            alpha *= 0.99
            
        # Save best state if valid (energy approx 0)
        if total_energy < 1e-6:
            best_centers = centers.copy()
            best_radii = radii.copy()
            best_sum = np.sum(radii)
        elif step > 200:
             # If we have been trying for a while and still high energy, maybe r is too big
             # But with the cooling, we might eventually settle.
             # For now, just continue.
             pass

    # Final validation check and cleanup
    # The simulation might leave slight overlaps due to discrete steps.
    # We can run a few steps with r fixed to clean up.
    final_r = best_radii[0]
    # Run local optimization with fixed r to ensure validity
    # Using a simple repulsive force relaxation
    
    centers_clean = best_centers.copy()
    radii_clean = best_radii.copy()
    
    # 50 refinement steps
    for _ in range(100):
        forces = np.zeros_like(centers_clean)
        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers_clean[i] - centers_clean[j]) ** 2))
                if dist < 2 * final_r:
                    overlap = 2 * final_r - dist
                    if dist > 1e-9:
                        force_vec = (overlap / dist) * (centers_clean[i] - centers_clean[j])
                        forces[i] += force_vec
                        forces[j] -= force_vec
        
        # Boundary
        # Left
        viol = final_r - centers_clean[:, 0]
        mask = viol > 0
        forces[mask, 0] += viol[mask]
        # Right
        viol = centers_clean[:, 0] - (1.0 - final_r)
        mask = viol > 0
        forces[mask, 0] -= viol[mask]
        # Bottom
        viol = final_r - centers_clean[:, 1]
        mask = viol > 0
        forces[mask, 1] += viol[mask]
        # Top
        viol = centers_clean[:, 1] - (1.0 - final_r)
        mask = viol > 0
        forces[mask, 1] -= viol[mask]
        
        centers_clean += forces * 0.5
        np.clip(centers_clean, 0, 1, out=centers_clean)
        
    # Final check
    # If validation fails, we might need to shrink radius slightly.
    # But let's trust the process.
    # To be safe, verify and shrink if needed.
    
    # Validation check (local)
    def is_valid(c, r_val):
        # Boundary
        if np.any(c[:, 0] - r_val < -1e-9) or np.any(c[:, 0] + r_val > 1 + 1e-9):
            return False
        if np.any(c[:, 1] - r_val < -1e-9) or np.any(c[:, 1] + r_val > 1 + 1e-9):
            return False
        # Overlap
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                if d < 2 * r_val - 1e-9:
                    return False
        return True

    # If the best found is not valid, shrink radius a bit
    if not is_valid(best_centers, best_radii[0]):
        # Try to shrink until valid
        r_test = best_radii[0]
        while not is_valid(best_centers, r_test) and r_test > 0:
            r_test *= 0.99
            # Re-optimize positions for this smaller r? 
            # Or just shrink radius?
            # If we just shrink radius, overlaps disappear.
            # But centers might be close to boundaries.
            # Just shrinking radii is safe for validity.
        best_radii[:] = r_test
    
    # Wait, if I just shrink radii, I don't move centers. 
    # The centers from 'best_centers' might be optimal for a slightly larger r.
    # But if overlaps exist, shrinking r resolves them.
    # So this is a valid fallback.
    
    # However, to maximize sum, we want the largest valid r.
    # Let's try a binary search or small adjustment for the final r.
    
    # Let's re-verify the final state.
    # The 'best_centers' and 'best_radii' were stored when energy was low.
    # They should be valid.
    
    # Let's return the cleaned up version
    final_centers = centers_clean
    final_radii = radii_clean
    
    # One last check on the cleaned centers
    # If cleaning moved things, radii might need check.
    # Actually, radii_clean was fixed at final_r.
    # We should verify.
    
    # Let's do a final robust check and shrink if necessary
    r_final = final_radii[0]
    while not is_valid(final_centers, r_final) and r_final > 0.01:
        r_final -= 0.001
        # If we shrink, we don't move centers, so validity improves.
        
    final_radii[:] = r_final
    
    # If the sum is too low (e.g. 0), fallback to grid
    if np.sum(final_radii) < 1.0:
        # Fallback to simple grid
        # 5x5 grid, r=0.1 is usually valid? 
        # 5 circles width 1.0. r=0.1.
        # Centers at 0.1, 0.3, 0.5, 0.7, 0.9.
        # Overlaps? No.
        # 25 circles. 26th?
        # Maybe 6x5 grid, r=0.08?
        # Let's just return the simulation result, it should be better.
        pass

    return final_centers, final_radii, np.sum(final_radii)
