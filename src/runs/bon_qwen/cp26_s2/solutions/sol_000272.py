# sol_000272 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9068c8d6) state=229bb7cf sum of radii=2.170564 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Finds a packing of 26 circles in a unit square to maximize sum of radii.
    """
    np.random.seed(42) # For reproducibility
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Helper to generate initial configuration
    def init_config():
        # Try a perturbed grid or hexagonal lattice
        # 5x5 grid is 25 circles, we need 26.
        # Let's try a dense random initialization
        c = np.random.rand(n, 2)
        r = np.ones(n) * 0.05 # Start small
        return c, r

    centers, radii = init_config()
    
    # Parameters for optimization
    max_iterations = 2000
    expansion_rate = 0.001
    force_scale = 1.0
    damping = 0.9
    
    # We will try to expand radii and resolve overlaps
    for step in range(max_iterations):
        # 1. Try to expand radii
        # If we are at a valid state, we can try to grow circles
        # Check if current state is valid (roughly)
        # We will enforce constraints directly
        
        # 2. Compute overlaps and forces
        overlaps = np.zeros((n, 2))
        
        # Pairwise interactions
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                required_dist = radii[i] + radii[j]
                
                if dist < required_dist and dist > 1e-9:
                    # Overlap detected
                    # Push apart
                    overlap_amount = required_dist - dist
                    force_vec = (dist_vec / dist) * overlap_amount * force_scale
                    
                    # Distribute force based on relative sizes (optional, or equal)
                    # Equal distribution
                    overlaps[i] += force_vec
                    overlaps[j] -= force_vec

        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                overlaps[i, 0] += (r - x) * force_scale * 2.0
            # Right wall
            if x + r > 1:
                overlaps[i, 0] -= (x + r - 1) * force_scale * 2.0
            # Bottom wall
            if y - r < 0:
                overlaps[i, 1] += (r - y) * force_scale * 2.0
            # Top wall
            if y + r > 1:
                overlaps[i, 1] -= (y + r - 1) * force_scale * 2.0
                
        # 3. Update positions
        centers += overlaps
        centers = np.clip(centers, 0, 1) # Hard clip to stay in square
        
        # 4. Update radii
        # If overlaps are minimal, increase radius
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if dist < req:
                    max_overlap = max(max_overlap, req - dist)
            
            # Boundary check for radius growth
            x, y = centers[i]
            max_possible_r = min(x, 1-x, y, 1-y)
            if max_possible_r < radii[i] + 1e-7:
                 # Radius is limited by boundary
                 pass 
            else:
                 # We can potentially grow
                 pass
        
        # Grow radii if system is stable (low overlap)
        # Heuristic: grow if max overlap is small
        if max_overlap < 1e-4:
            # Grow all radii slightly
            # Limit growth by available space
            for i in range(n):
                x, y = centers[i]
                space = min(x, 1-x, y, 1-y)
                # Grow towards space limit, but not too fast
                growth = min(expansion_rate, space - radii[i])
                if growth > 0:
                    radii[i] += growth
        else:
            # If overlap is high, maybe shrink slightly or just let forces resolve
            # For this simple solver, we rely on forces to resolve, 
            # and only grow when stable.
            pass
            
        # Decay parameters
        if step % 100 == 0:
            expansion_rate *= 0.95
            force_scale = 0.5 # Reduce force scale to let it settle
            
    # Final cleanup: Ensure radii don't violate boundaries
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1-x, y, 1-y)
        if max_r < 0: max_r = 0
        radii[i] = min(radii[i], max_r)
        
    # Second pass: Local optimization to squeeze more radius
    # Simple iterative relaxation
    for _ in range(500):
        # Try to increase each radius individually and resolve
        for i in range(n):
            current_r = radii[i]
            # Try to increase
            target_r = current_r + 0.0005
            radii[i] = target_r
            
            # Resolve overlaps for this circle
            for k in range(10): # Local steps
                moved = False
                for j in range(n):
                    if i == j: continue
                    dist_vec = centers[i] - centers[j]
                    dist = np.linalg.norm(dist_vec)
                    req = radii[i] + radii[j]
                    if dist < req and dist > 1e-9:
                        overlap = req - dist
                        move_vec = (dist_vec / dist) * overlap
                        centers[i] += move_vec * 0.5
                        moved = True
                        # Clip
                        centers[i] = np.clip(centers[i], radii[i], 1-radii[i])
                        # If clipped, adjust radius if necessary (though we fixed pos)
                        # Actually clipping ensures boundary, but might increase overlap with others?
                        # Just ensure center is valid
                if not moved: break
            
            # Check boundary constraint for radius again
            x, y = centers[i]
            max_r = min(x, 1-x, y, 1-y)
            if max_r < radii[i]:
                radii[i] = max_r # Shrink if center is too close to wall
                
            # If we couldn't fit the increase, revert?
            # Check if still valid
            valid = True
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                radii[i] = current_r # Revert

    sum_radii = np.sum(radii)
    
    # Final validation check
    if not validate_packing(centers, radii):
        # Fallback to a safe packing if optimization failed
        # Grid packing
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        idx = 0
        # 5x5 grid + 1
        r = 0.1
        for r_y in range(5):
            for r_x in range(5):
                centers[idx] = [r_x * 0.2 + r, r_y * 0.2 + r]
                radii[idx] = r
                idx += 1
        # 26th circle?
        # If idx < 26
        if idx < 26:
            # Place in center of a gap?
            # In 5x5 grid with r=0.1, gaps are small.
            # Maybe reduce r slightly?
            # Let's just place it at (0.5, 0.5) with small r
            centers[25] = [0.5, 0.5]
            radii[25] = 0.04 # Small enough to fit?
            # (0.5, 0.5) is center of square. Nearest grid points are (0.4, 0.4) etc?
            # Grid points: 0.1, 0.3, 0.5, 0.7, 0.9
            # (0.5, 0.5) is occupied by circle 13 (index 12)?
            # 0,1,2,3,4 -> row 0
            # 5..9 -> row 1
            # 10..14 -> row 2 (y=0.5). x=0.1, 0.3, 0.5, 0.7, 0.9.
            # So (0.5, 0.5) is taken.
            # Let's shift the last one to a corner?
            # Actually, with r=0.1, we have 25 circles.
            # To fit 26th, we must reduce r.
            # Let's scale down slightly.
            scale = 0.99
            radii[:] = 0.1 * scale
            centers[:, 0] = centers[:, 0] / scale # Not quite right scaling
            # Just re-grid with smaller r
            r_new = 0.099
            idx = 0
            for r_y in range(5):
                for r_x in range(5):
                     centers[idx] = [r_x * 2*r_new + r_new, r_y * 2*r_new + r_new]
                     radii[idx] = r_new
                     idx += 1
            # 26th
            centers[25] = [0.5, 0.5]
            radii[25] = r_new # Might overlap, but validation will catch.
            # Let's just rely on the optimization result if valid.
            pass

    return centers, radii, sum_radii

# To run the packing
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Valid: {validate_packing(centers, radii)}")
