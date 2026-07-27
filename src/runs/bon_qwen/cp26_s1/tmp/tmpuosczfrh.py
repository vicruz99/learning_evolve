import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    best_r = 0.0
    best_centers = None
    
    # Function to calculate the minimum clearance (radius limit) for a given set of centers
    def get_max_radius(centers):
        min_dist = 1.0
        
        # Check boundary distances
        # Radius must be <= distance to any wall
        for i in range(n_circles):
            x, y = centers[i]
            d_x = min(x, 1 - x)
            d_y = min(y, 1 - y)
            min_dist = min(min_dist, d_x, d_y)
            
        # Check pairwise distances
        # 2*radius <= distance between centers => radius <= dist/2
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                min_dist = min(min_dist, dist / 2.0)
                
        return min_dist

    # Helper to initialize centers in a hexagonal pattern
    def init_hexagonal(n, rows, cols_per_row):
        # Approximate radius for hex packing
        # Area density pi/sqrt(12) ~ 0.9069
        # n * pi * r^2 / 0.9069 <= 1 => r ~ sqrt(0.9069 / (n * pi))
        est_r = np.sqrt(0.9069 / (n * np.pi))
        
        # Adjust spacing based on estimated radius
        # Hex lattice: x spacing 2r, y spacing sqrt(3)r
        # We need to fit into [0,1]
        
        centers = np.zeros((n, 2))
        idx = 0
        
        # Generate lattice points
        points = []
        for r_idx in range(rows):
            num_circles = cols_per_row[r_idx]
            for c_idx in range(num_circles):
                x = c_idx * 2.0 * est_r
                y = r_idx * np.sqrt(3) * est_r
                if r_idx % 2 == 1:
                    x += est_r # Shift odd rows
                points.append([x, y])
        
        if len(points) < n:
            # Fallback to random if pattern doesn't match n
            centers = np.random.uniform(0.1, 0.9, (n, 2))
        else:
            # Scale and translate to fit in unit square
            # We want to maximize the radius, so we scale the configuration
            # The configuration has some bounding box. We scale it to fit [0,1] with margin.
            pts = np.array(points[:n])
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            
            # Current size
            size_x = max_x - min_x
            size_y = max_y - min_y
            size = max(size_x, size_y)
            
            # We want the scaled size + 2*est_r to be <= 1?
            # Actually, just normalize to [0,1] roughly and let optimizer fix it.
            # Simple scaling to [0.1, 0.9]
            if size > 1e-9:
                scale = 0.8 / size
                pts = (pts - min_x) * scale + 0.1
            else:
                pts = np.random.uniform(0.1, 0.9, (n, 2))
                
            centers = pts

        return centers

    # Force-directed optimization for a fixed radius r
    # Tries to push circles apart if they overlap
    def relax_packing(centers, r, steps=500):
        n = len(centers)
        current_centers = centers.copy()
        
        # Learning rate
        lr = 0.01
        
        for step in range(steps):
            forces = np.zeros_like(current_centers)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    diff = current_centers[i] - current_centers[j]
                    dist = np.sqrt(np.sum(diff**2))
                    if dist < 1e-9:
                        # Avoid division by zero, push random
                        diff = np.random.randn(2) * 0.01
                        dist = 1e-9
                    
                    # Overlap amount
                    overlap = 2.0 * r - dist
                    if overlap > 0:
                        # Push apart
                        force = (diff / dist) * overlap
                        forces[i] += force
                        forces[j] -= force
            
            # Wall repulsion
            for i in range(n):
                x, y = current_centers[i]
                # Left wall
                if x < r:
                    forces[i, 0] += (r - x)
                # Right wall
                if x > 1 - r:
                    forces[i, 0] -= (x - (1 - r))
                # Bottom wall
                if y < r:
                    forces[i, 1] += (r - y)
                # Top wall
                if y > 1 - r:
                    forces[i, 1] -= (y - (1 - r))
            
            # Apply forces
            current_centers += lr * forces
            
            # Clamp to valid region loosely to prevent explosion
            current_centers = np.clip(current_centers, 0.0, 1.0)
            
            # Reduce learning rate
            lr *= 0.99
            
        return current_centers

    # Strategy: Try multiple initial configurations and optimize
    # We aim for r ~ 0.101 to 0.105
    
    # Candidate configurations
    configs = []
    
    # 1. Hexagonal 5 rows
    # Rows: 5, 5, 5, 5, 5 (25) + 1 somewhere?
    # Let's try 5, 5, 5, 5, 6? Or 6, 5, 5, 5, 5?
    # Width of 6 circles is large.
    # Let's try a perturbed grid
    
    # Config 1: Random
    configs.append(np.random.uniform(0.1, 0.9, (n_circles, 2)))
    
    # Config 2: Hexagonal 5x5 + 1
    # 5 rows of 5 is 25. Add 1 in middle.
    # Generate 5x5 grid first
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    cx, cy = np.meshgrid(grid_x, grid_y)
    grid_centers = np.column_stack((cx.ravel(), cy.ravel()))
    # Add center point if not present (0.5, 0.5 is in grid)
    # Remove center and add perturbed points?
    # Just add a point at (0.5, 0.5) and remove one?
    # Actually, just take 26 points from a larger grid or pattern.
    
    # Let's generate a dense hex pattern
    # 6 rows?
    # Row 0: 5
    # Row 1: 5
    # Row 2: 5
    # Row 3: 5
    # Row 4: 5
    # Row 5: 1
    # This is hard to fit.
    
    # Let's just use the relax_packing function to find max r.
    # We can binary search r.
    
    low_r = 0.05
    high_r = 0.15
    best_r = 0.0
    best_centers = None
    
    # Number of binary search steps
    for _ in range(20):
        mid_r = (low_r + high_r) / 2.0
        
        # Try to find a valid configuration for mid_r
        # Use a few random restarts for robustness
        success = False
        
        # We can start from a previous best configuration if it exists
        # But for binary search, we need to check feasibility.
        # Feasibility is hard to check exactly, but relaxation helps.
        
        # Try 3 random starts
        for restart in range(3):
            if restart == 0 and best_centers is not None:
                centers = best_centers.copy()
            else:
                centers = np.random.uniform(mid_r, 1 - mid_r, (n_circles, 2))
            
            # Relax
            relaxed_centers = relax_packing(centers, mid_r, steps=200)
            
            # Check validity
            # Check overlaps
            valid = True
            # Boundary check
            if np.any(relaxed_centers < mid_r) or np.any(relaxed_centers > 1 - mid_r):
                valid = False # Simple check, relax_packing clamps but might still be slightly out or stuck
            
            # Overlap check
            # We need to check if min_dist >= 2*mid_r
            min_d = 1.0
            for i in range(n_circles):
                # Boundary
                d_b = min(relaxed_centers[i,0], 1-relaxed_centers[i,0], 
                          relaxed_centers[i,1], 1-relaxed_centers[i,1])
                min_d = min(min_d, d_b)
                for j in range(i+1, n_circles):
                    dist = np.sqrt(np.sum((relaxed_centers[i] - relaxed_centers[j])**2))
                    min_d = min(min_d, dist)
            
            # Effective radius achievable
            achievable_r = min_d / 2.0
            
            if achievable_r >= mid_r - 1e-5:
                success = True
                best_r = achievable_r
                best_centers = relaxed_centers
                break
        
        if success:
            low_r = mid_r
        else:
            high_r = mid_r

    # Final refinement
    # Use the best centers found and try to squeeze more
    if best_centers is not None:
        # Run a few more steps with higher precision or different method
        # We can try to optimize the minimum distance directly
        # But simple relaxation with slightly higher r might work
        
        # Try to increase r slightly from best_r
        current_r = best_r
        current_centers = best_centers
        
        for _ in range(50):
            # Increase r by small amount
            test_r = current_r * 1.001
            relaxed = relax_packing(current_centers, test_r, steps=100)
            
            # Check validity
            min_d = 1.0
            for i in range(n_circles):
                d_b = min(relaxed[i,0], 1-relaxed[i,0], relaxed[i,1], 1-relaxed[i,1])
                min_d = min(min_d, d_b)
                for j in range(i+1, n_circles):
                    dist = np.sqrt(np.sum((relaxed[i] - relaxed[j])**2))
                    min_d = min(min_d, dist)
            
            if min_d / 2.0 >= test_r - 1e-5:
                current_r = test_r
                current_centers = relaxed
            else:
                break
        
        best_r = current_r
        best_centers = current_centers

    # If binary search didn't find good solution, try a specific heuristic packing
    # Hexagonal packing attempt
    # 26 circles.
    # Try to fit in 5 rows.
    # Row lengths: 5, 5, 5, 5, 6 (26)
    # Width for 6 circles: 5*2r. Height for 5 rows: 4*sqrt(3)r.
    # Constraints: 10r + 2r <= 1 => 12r <= 1 => r <= 0.0833
    # Height: 4*1.732r + 2r = 8.928r <= 1 => r <= 0.112
    # So width is bottleneck. r <= 0.083. Sum = 2.16.
    # 5,5,5,5,5 (25) + 1 in gap.
    # 5x5 grid r=0.1. Sum=2.5.
    # We can definitely do better than 0.083.
    
    # The binary search approach with relaxation should find something near 0.10.
    
    # Ensure radii are non-negative and valid
    if best_r < 1e-4:
        # Fallback to a valid packing
        best_r = 0.05
        best_centers = np.random.uniform(best_r, 1-best_r, (n_circles, 2))
        # Simple grid
        step = 0.8 / 5
        cx = np.linspace(0.1, 0.9, 6)[:5]
        cy = np.linspace(0.1, 0.9, 5)
        pts = []
        for x in cx:
            for y in cy:
                pts.append([x,y])
        # pts has 25. Add one.
        pts.append([0.5, 0.15])
        best_centers = np.array(pts[:26])

    # Calculate final radii
    # To be safe and valid, we can set all radii to best_r
    # But maybe we can vary them slightly to increase sum?
    # The problem allows variable radii.
    # If we have a configuration for radius R, we can set r_i = R.
    # Can we increase sum by making some larger?
    # If we have slack, yes. But packing is usually tight.
    # Let's just use equal radii = best_r.
    
    radii = np.full(n_circles, best_r)
    sum_radii = np.sum(radii)
    
    # Final validation check
    # If invalid, shrink slightly
    for _ in range(10):
        # Check overlaps
        overlap_found = False
        for i in range(n_circles):
            x, y = best_centers[i]
            r = radii[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                overlap_found = True
                break
            for j in range(i+1, n_circles):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-9:
                    overlap_found = True
                    break
            if overlap_found: break
        
        if not overlap_found:
            break
        
        # Shrink
        best_r *= 0.99
        radii = np.full(n_circles, best_r)
        sum_radii = np.sum(radii)

    return best_centers, radii, sum_radii