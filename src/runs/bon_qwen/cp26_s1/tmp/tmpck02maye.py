import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    square_size = 1.0
    
    # --- Initialization: Hexagonal Grid ---
    # We want to fit 26 circles. A hexagonal packing is dense.
    # Let's try to estimate a good starting radius and layout.
    # For N=26, a 6-row hexagonal packing or similar is good.
    # Let's place them in a grid first and then optimize.
    
    # Strategy: Place circles in a perturbed hexagonal lattice
    # Estimated radius for 26 circles is around 0.101.
    # Let's start with r = 0.08 to ensure no overlaps, then grow.
    
    initial_r = 0.05
    centers = np.zeros((n_circles, 2))
    radii = np.full(n_circles, initial_r)
    
    # Generate hexagonal grid positions
    # Rows and columns to approximate 26 circles
    # 5 rows: 5, 5, 5, 5, 6? Or 6, 5, 6, 5, 4?
    # Let's just fill a grid and trim/adjust
    
    # Approximate spacing
    # If r ~ 0.1, 2r ~ 0.2. 1/0.2 = 5.
    # 5x5 grid = 25 circles. We need 26.
    # Let's try a layout with 6 rows to utilize space better or a specific pattern.
    
    # Let's create a list of (row, col) for a hexagonal lattice
    # Row y = r + row_idx * r * sqrt(3)
    # Col x = r + col_idx * 2r + (row_idx % 2) * r
    
    # We need to determine number of rows and cols to get ~26 circles.
    # Let's try to fit them in a 6x5 area roughly?
    # 6 rows, 5 cols would be 30 circles. We need to remove 4.
    # Or 5 rows, 6 cols?
    
    # Let's generate a larger grid and pick 26 points that are well distributed,
    # or just generate exactly 26 points.
    
    # Configuration: 
    # Row 0: 5 circles
    # Row 1: 5 circles (offset)
    # Row 2: 5 circles
    # Row 3: 5 circles (offset)
    # Row 4: 6 circles?
    # Total 26.
    
    # Let's calculate positions for a specific count configuration.
    # To maximize density, rows with offset usually fit fewer circles for same width.
    # Even rows (0, 2, 4): start at x=r. Capacity k: width 2kr <= 1 => k <= 1/(2r).
    # Odd rows (1, 3): start at x=2r. Capacity k: width (2k+1)r <= 1 => k <= (1/r - 1)/2.
    
    # If we target r ~ 0.1:
    # Even row capacity: 5 (width 1.0)
    # Odd row capacity: 4 (width 0.9) or 5? 5 requires width 1.1 (fail).
    # So 5, 4, 5, 4, 5 = 23 circles.
    # We need 26. This implies we need more rows or a different arrangement.
    # Maybe 6 rows?
    # 6 rows height: 2r + 5*r*sqrt(3) ~ 10.66r. If r=0.094, height=1.0.
    # With r=0.094, width capacity Even: floor(1/0.188) = 5.
    # Odd: floor((1/0.094 - 1)/2) = floor(4.8) = 4.
    # Pattern 5, 4, 5, 4, 5, 4 = 27 circles.
    # This fits 26 circles easily with r=0.094.
    # Sum = 26 * 0.094 = 2.44.
    # We want sum ~ 2.636 => r ~ 0.101.
    # At r=0.101, 6 rows height = 10.66 * 0.101 = 1.07 > 1. Fails.
    # 5 rows height = 8.928 * 0.101 = 0.90 < 1. OK.
    # Width at r=0.101:
    # Even capacity: floor(1/0.202) = 4.
    # Odd capacity: floor((9.9-1)/2) = 4.
    # 5 rows of 4 = 20 circles. Not enough.
    
    # Conclusion: Standard axis-aligned hexagonal lattice cannot reach 26 circles with r=0.101.
    # However, the optimization might find a non-lattice packing or rotate the lattice.
    # Or perhaps the "sum of radii" allows unequal circles which can pack better?
    # Or maybe my capacity estimate is too strict (we can squeeze a bit).
    
    # Let's initialize with a dense random packing or a slightly perturbed grid
    # and let the optimizer do the work.
    
    # Better initialization:
    # Place circles on a 5x5 grid (25 circles) + 1 circle in the center gap?
    # Grid 5x5: centers at (0.2, 0.2), (0.2, 0.4), ... (0.8, 0.8) with r=0.1.
    # Actually 5 circles in [0,1] with spacing 0.2?
    # Centers: 0.1, 0.3, 0.5, 0.7, 0.9.
    # Radius 0.1 fits exactly (0.0 to 0.2, ..., 0.8 to 1.0).
    # 25 circles of radius 0.1. Sum = 2.5.
    # Where to put the 26th?
    # The gaps are curvilinear triangles.
    # Center of square (0.5, 0.5) is occupied by a circle.
    # Gaps are at (0.2, 0.2) type positions? No, those are centers.
    # Gaps are at (0.4, 0.4) relative to centers?
    # Distance from (0.3, 0.3) to (0.5, 0.5) is sqrt(0.08) approx 0.282.
    # Radius 0.1 + 0.1 = 0.2. Gap is 0.082.
    # Can we fit a circle of radius ~0.04?
    # Sum would be 2.5 + 0.04 = 2.54.
    # We need 2.636.
    # So we need to expand the radii.
    # If we expand radii from 0.1, the 25 circles will overlap or hit boundaries.
    # We must move them apart.
    
    # Let's initialize with 26 circles in a "random" valid configuration
    # that is dense, then optimize.
    # A good starting point is to pack them tightly.
    
    # Let's try a specific dense packing known for N=26 if possible,
    # or just use a force-directed layout starting from a compact grid.
    
    # Layout: 6 rows.
    # Row counts: 5, 5, 5, 5, 3, 3? No.
    # Let's just generate points in a hexagonal pattern with small radius
    # and then grow.
    
    # Let's try to fit 26 circles with r=0.1 in a square by perturbation.
    # Start with 5x5 grid (25 circles) at r=0.1.
    # Add 1 circle at (0.5, 0.5) with small radius, then push everything.
    # Actually, (0.5, 0.5) is a center in 5x5 grid.
    # Let's shift the grid slightly to make room.
    
    # Let's use a more robust initialization:
    # Fill the square with a Poisson disk sampling or just a perturbed grid.
    
    rng = np.random.default_rng(42)
    
    # Start with a grid of 6x5 = 30 positions, remove 4 randomly, keep 26.
    # Grid spacing 1/5 = 0.2.
    # Points: x in [0.1, 0.3, 0.5, 0.7, 0.9], y in [0.1, ..., 0.9]
    # Add an offset row?
    
    # Let's try a 6x5 hexagonal-ish layout.
    # 6 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 5 circles
    # Row 5: 1 circle (center?)
    # Total 26.
    
    # Let's construct centers manually for a good start.
    # We want them roughly uniformly distributed.
    
    # Method: Spiral or Grid with noise.
    # Grid 5x5 is 25 points.
    # We need 1 more.
    # Let's place 26 points using a low-discrepancy sequence (Sobol) or just grid + center.
    # But grid + center overlaps.
    
    # Let's use a hexagonal packing of 27 circles (which we know fits with r~0.094)
    # and remove the one with smallest radius potential? No, all equal.
    # Just remove one circle from the 27-circle hexagonal packing.
    # This leaves a valid packing of 26 circles with r=0.094.
    # Then we optimize to increase r.
    
    # Construction of 27 circles hexagonal packing:
    # 6 rows.
    # Even rows (0, 2, 4): 5 circles.
    # Odd rows (1, 3, 5): 4 circles.
    # Wait, 5+4+5+4+5+4 = 27.
    # Let's build this.
    
    # Parameters for initial hexagonal packing
    r_init = 0.09 # Safe initial radius
    
    # Row y-coordinates
    # y_k = r + k * r * sqrt(3)
    # We have 6 rows (k=0..5)
    
    rows_y = []
    for k in range(6):
        y = r_init + k * r_init * np.sqrt(3)
        rows_y.append(y)
        
    # Construct centers
    current_centers = []
    
    for k in range(6):
        y = rows_y[k]
        is_odd = (k % 2 == 1)
        
        if is_odd:
            # Odd row: 4 circles. Shifted by r.
            # Centers at 2r, 4r, 6r, 8r?
            # Width check: (2*4 + 1)r = 9r. 9*0.09 = 0.81 <= 1. OK.
            # Let's center them in the square?
            # Current span: from r to 9r? No.
            # First center 2r, last center 2r + 3*2r = 8r.
            # Extent [r, 9r].
            # To center in [0,1], we can shift x by (1 - 9r)/2.
            
            n_circ = 4
            # Base x positions for 4 circles in odd row (relative to left wall logic)
            # 2r, 4r, 6r, 8r
            base_xs = np.array([2, 4, 6, 8]) * r_init
            
            # Calculate shift to center
            span = (2 * n_circ + 1) * r_init # 9r
            shift = (1.0 - span) / 2.0
            
            xs = base_xs + shift
            for x in xs:
                current_centers.append([x, y])
        else:
            # Even row: 5 circles.
            # Centers at r, 3r, 5r, 7r, 9r.
            # Span 10r.
            n_circ = 5
            base_xs = np.array([1, 3, 5, 7, 9]) * r_init
            span = 2 * n_circ * r_init # 10r
            shift = (1.0 - span) / 2.0
            xs = base_xs + shift
            for x in xs:
                current_centers.append([x, y])
                
    # We have 27 circles. Remove the last one to get 26.
    # Or remove the one that is most constrained? 
    # Just remove the last one added (center of last row? No, last row is odd, 4 circles).
    # The last added circle is in row 5 (odd), index 3.
    # It's a valid circle. Removing it leaves 26 circles.
    
    if len(current_centers) > 26:
        # Remove a circle from the middle to keep distribution good?
        # Removing from end is fine.
        current_centers = current_centers[:26]
    else:
        # Should not happen based on math, but safety
        pass
        
    centers = np.array(current_centers)
    radii = np.full(26, r_init)
    
    # --- Optimization ---
    # We will iteratively increase radii and resolve collisions.
    
    # Optimization parameters
    max_iter = 2000
    growth_rate = 1.001 # Multiply radii by this
    repulsion_strength = 10.0
    damping = 0.1
    step_size = 0.001
    
    # To maximize sum of radii, we can treat radii as variables.
    # But a simpler heuristic for packing is:
    # 1. Fix radii, move centers to reduce overlaps.
    # 2. Increase radii slightly.
    # Repeat.
    
    # However, we want to maximize sum(r_i).
    # Let's use a simple energy minimization with a Lagrangian-like approach or just repulsion.
    # Energy = Sum of squared overlaps + Sum of squared boundary violations.
    # We want to minimize Energy while keeping radii large?
    # No, we want to find the configuration where radii are maximal.
    
    # Algorithm:
    # Loop:
    #   1. Calculate forces on centers due to overlaps and boundaries.
    #   2. Move centers to reduce forces (gradient descent on energy).
    #   3. If energy is low (no overlaps), increase radii slightly.
    #   4. Repeat.
    
    # Energy function
    # E = sum_{i<j} max(0, r_i + r_j - dist_ij)^2 
    #   + sum_i max(0, r_i - x_i)^2 + max(0, x_i + r_i - 1)^2 + ...
    
    # We will perform coordinate descent / gradient descent on (centers, radii).
    # But radii only increase.
    
    # Let's implement a simple iterative solver.
    
    centers = centers.copy()
    radii = radii.copy()
    
    # Precompute indices
    n = len(centers)
    idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for iteration in range(max_iter):
        # Check if we are stuck or done? 
        # We run fixed iterations.
        
        # 1. Calculate gradients/forces for centers
        # We want to minimize overlaps.
        # Force on i from j: if overlap, push i away from j.
        # Overlap amount: r_i + r_j - dist
        # Force vector direction: (c_i - c_j) / dist
        # Magnitude: overlap * strength
        
        forces = np.zeros_like(centers)
        energy = 0.0
        
        # Overlap forces
        for i, j in idx_pairs:
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-9: dist = 1e-9 # Avoid div by zero
            
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                overlap = sum_r - dist
                # Penalty energy
                energy += overlap**2
                
                # Force direction: push apart
                # Force on i: +overlap * (diff/dist)
                # Force on j: -overlap * (diff/dist)
                force_vec = (diff / dist) * overlap
                forces[i] += force_vec * repulsion_strength
                forces[j] -= force_vec * repulsion_strength
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                violation = r - x # amount penetrated? No, x-r < 0 => r-x > 0
                # Actually, constraint is x >= r. Violation = r - x.
                viol = r - x
                energy += viol**2
                forces[i, 0] += viol * repulsion_strength # Push right
            
            # Right wall
            if x + r > 1:
                viol = x + r - 1
                energy += viol**2
                forces[i, 0] -= viol * repulsion_strength # Push left
                
            # Bottom wall
            if y - r < 0:
                viol = r - y
                energy += viol**2
                forces[i, 1] += viol * repulsion_strength # Push up
                
            # Top wall
            if y + r > 1:
                viol = y + r - 1
                energy += viol**2
                forces[i, 1] -= viol * repulsion_strength # Push down
        
        # Apply forces to move centers
        # Update centers = centers + step * forces
        # Use a decaying step size or adaptive
        
        # We also want to increase radii if energy is low.
        if energy < 1e-6:
            # No significant overlaps, try to grow
            radii *= (1.0 + 0.0005) # Small growth
            # Occasionally add random perturbation to escape local minima?
            if iteration % 100 == 0:
                centers += np.random.normal(0, 0.001, centers.shape)
        else:
            # High energy, move centers to resolve
            # Step size proportional to energy? Or fixed small step.
            # If energy is high, we need to move more?
            # But too large step causes oscillation.
            centers += step_size * forces
            # Clip centers to stay inside roughly (optional, forces handle it)
            # But let's ensure they don't go wild
            centers = np.clip(centers, 0, 1)
            
        # Occasionally reduce step size?
        # Actually, if energy is low, we grow radii, which increases energy next step.
        # If energy is high, we move centers.
        
        # Random restarts or perturbations might help, but let's rely on forces.
        
    # Final validation and cleanup
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0)
    
    # Ensure centers are in [0, 1]
    centers = np.clip(centers, 0, 1)
    
    # One last check to fix any minor overlaps due to numerical errors
    # by slightly reducing radii if needed
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Boundary
        r = min(r, x, 1-x, y, 1-y)
        
        # Overlaps
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            sum_r = r + radii[j]
            if dist < sum_r:
                # Reduce r to fit
                # We want to preserve as much area as possible, but just need valid packing.
                # Reduce r equally? Or just r_i?
                # Let's reduce r_i just enough to touch.
                required_r = dist - radii[j]
                if required_r < r:
                    r = max(0, required_r - 1e-9) # small buffer
        radii[i] = r
        
    sum_radii = np.sum(radii)
    
    # Validate
    # (Implicitly validated by the reduction step, but let's be sure)
    # The reduction step ensures validity.
    
    return centers, radii, sum_radii