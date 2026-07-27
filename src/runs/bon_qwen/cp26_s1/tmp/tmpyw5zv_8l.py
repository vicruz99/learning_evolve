import numpy as np
import scipy.optimize as opt
import math

def run_packing():
    """
    Returns (centers, radii, sum_radii) for a packing of 26 circles in a unit square.
    """
    n_circles = 26
    
    # Helper function to calculate constraints and objective
    # We want to maximize sum(radii), so we minimize -sum(radii)
    # Constraints:
    # 1. Boundary: r <= x <= 1-r, r <= y <= 1-r
    # 2. Non-overlap: dist(p_i, p_j) >= r_i + r_j
    
    # Since handling many inequality constraints is hard, we use a penalty method.
    # We optimize over a vector [x1, y1, r1, x2, y2, r2, ...]
    
    best_sum_radii = -1.0
    best_params = None
    
    # Number of restarts
    n_restarts = 20
    
    for seed in range(n_restarts):
        # Initialize positions
        # Strategy: Hexagonal packing or Grid
        # Let's try a grid perturbed randomly
        np.random.seed(seed)
        
        # Initial guess: Grid
        # 5x5 grid has 25 circles. We need 26.
        # Let's place 26 circles in a roughly hexagonal pattern or random dense packing
        
        # Try to generate a hexagonal lattice subset
        centers_init = []
        radii_init = []
        
        # Hexagonal packing parameters
        # Try to fit 26 circles
        # A 6-row pattern might work better? Or 5 rows with extra?
        # Let's just place them randomly inside [0.1, 0.9] and let optimizer sort it out
        # Or better: a specific structured init
        
        # Init 1: Grid 5x5 + 1 extra
        # Grid points
        gs = 5
        for i in range(gs):
            for j in range(gs):
                centers_init.append([0.1 + i * 0.2, 0.1 + j * 0.2])
        
        # 26th circle? Maybe center?
        # But (0.5, 0.5) is occupied.
        # Let's place 26th at (0.1, 0.1) with 0 radius? No.
        # Let's just place 26 random points
        
        if len(centers_init) < n_circles:
             # Fill remaining
             for _ in range(n_circles - len(centers_init)):
                 centers_init.append([0.5 + np.random.randn()*0.1, 0.5 + np.random.randn()*0.1])
        
        # Actually, let's just generate a dense random packing initialization
        # Or a specific lattice. 
        # Let's try a hexagonal lattice with spacing determined by density.
        # Area ~ 1/26. Radius ~ 0.1. Spacing ~ 0.2.
        
        centers = []
        y_curr = 0.1
        while len(centers) < n_circles:
            x_curr = 0.1
            row_len = 0
            shift = 0 if int(y_curr / 0.1) % 2 == 0 else 0.1 # Offset for hex
            # Actually hex offset is 0.1 (half spacing)
            # Spacing 0.2. Offset 0.1.
            
            while x_curr <= 0.9 and len(centers) < n_circles:
                centers.append([x_curr + shift, y_curr])
                x_curr += 0.2
                row_len += 1
            y_curr += 0.1732 # sqrt(3)/2 * 0.2
            
        # Trim to 26
        centers = centers[:n_circles]
        radii = [0.09] * n_circles # Initial radius
        
        # Reshape to optimization vector
        # [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.array([])
        for c, r in zip(centers, radii):
            x0 = np.append(x0, [c[0], c[1], r])
            
        # Bounds
        # x, y in [0, 1], r in [0, 0.5]
        # But tighter bounds might help: r <= 0.15
        bnds = []
        for _ in range(n_circles):
            bnds.extend([(0, 1), (0, 1), (0, 0.2)]) # x, y, r
            
        def objective(vars):
            # vars shape (3 * n_circles,)
            sum_r = 0.0
            penalty = 0.0
            
            # Reshape
            centers = np.zeros((n_circles, 2))
            radii = np.zeros(n_circles)
            
            for i in range(n_circles):
                idx = i * 3
                cx, cy, r = vars[idx], vars[idx+1], vars[idx+2]
                centers[i] = [cx, cy]
                radii[i] = r
                sum_r += r
                
                # Boundary penalties
                # We need r <= x <= 1-r => x-r >= 0, x+r <= 1
                # y-r >= 0, y+r <= 1
                # Penalty if violated
                margin = 1e-6
                if cx - r < margin:
                    penalty += (margin - (cx - r)) ** 2 * 1000
                if cx + r > 1 - margin:
                    penalty += ((cx + r) - (1 - margin)) ** 2 * 1000
                if cy - r < margin:
                    penalty += (margin - (cy - r)) ** 2 * 1000
                if cy + r > 1 - margin:
                    penalty += ((cy + r) - (1 - margin)) ** 2 * 1000

            # Overlap penalties
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2 * 1000 # Heavy penalty

            return -sum_r + penalty

        # Optimization
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bnds, 
                               options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
            
            if res.success or res.fun < 0: # fun is negative sum + penalty
                # Extract solution
                sol_centers = np.zeros((n_circles, 2))
                sol_radii = np.zeros(n_circles)
                for i in range(n_circles):
                    idx = i * 3
                    sol_centers[i] = [res.x[idx], res.x[idx+1]]
                    sol_radii[i] = res.x[idx+2]
                
                # Check validity
                if validate_packing(sol_centers, sol_radii):
                    s = np.sum(sol_radii)
                    if s > best_sum_radii:
                        best_sum_radii = s
                        best_params = (sol_centers, sol_radii)
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback to simple grid
        centers = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)])
        # Add one more?
        # Just duplicate one with 0 radius? No, must be valid.
        # Let's just return a valid small packing
        centers = centers[:25]
        radii = np.array([0.1]*25)
        # Need 26.
        # Let's shrink all to fit 26?
        # 26 circles of radius r. 5x5 grid can't fit 26.
        # Let's use a 5x6 grid? No.
        # Let's just return the best found or a valid default.
        # If optimization failed, return a safe valid packing.
        # 26 circles of radius 0.08 might fit?
        # 5x5 grid r=0.1. 26th circle?
        # Maybe reduce r to 0.095?
        r_safe = 0.095
        # 26 circles. 5x5 grid + 1?
        # Maybe 6x5 grid?
        # Let's just return 26 circles of small radius in a grid-like pattern
        # Or just the best_params if found.
        # If not found, we have a problem.
        # But with 20 restarts, it should find something.
        pass

    if best_params is not None:
        return best_params[0], best_params[1], best_sum_radii
    else:
        # Fallback
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        # Try to fit 26 circles of radius 0.08
        # 6 rows of 5? No, 30 circles.
        # 5 rows of 5 is 25.
        # 5 rows of 6 is 30.
        # Let's try 5 rows.
        # Row 1: 5 circles
        # Row 2: 5
        # ...
        # Maybe staggered?
        # Let's just place them randomly and scale down?
        # Or use the grid logic.
        # 26 circles.
        # Maybe a 5x5 grid with radius r, and one small circle?
        # If r=0.09, sum=2.5 + small.
        # Let's try r=0.1 for 25, and 0 for 1? No.
        # Let's try r=0.095 for 26?
        # 26 * pi * 0.095^2 approx 0.73.
        # Grid 5x5 requires width 1. 5*2r = 10r <= 1 => r <= 0.1.
        # So r=0.095 fits in 5x5 grid?
        # Width 5*2*0.095 = 0.95 <= 1. Yes.
        # But we need 26 circles.
        # 5x5 grid has 25 slots.
        # Where to put 26th?
        # Maybe a 5x6 grid? 5*2*0.095 = 0.95 width. 6*2*0.095 = 1.14 height. No.
        # Maybe 6x5? 6 width, 5 height.
        # Width 1.14, Height 0.95. No.
        # Maybe hexagonal packing of 26 circles?
        # Let's try to construct a valid packing manually for fallback.
        # 5 rows.
        # Row 1: 5 circles
        # Row 2: 5 circles
        # Row 3: 5 circles
        # Row 4: 5 circles
        # Row 5: 6 circles?
        # If we stagger, maybe we can fit 6 in a row?
        # With r=0.09, diameter 0.18.
        # 6 circles width 6*0.18 = 1.08. Too wide.
        # But staggered?
        # Row 1: 5 circles. x: 0.09, 0.27, 0.45, 0.63, 0.81. (Span 0.9)
        # Row 2: 6 circles?
        # Shifted by 0.09. x: 0.18, 0.36, 0.54, 0.72, 0.90, 1.08? No.
        # Maybe 5, 6, 5, 6?
        # Let's just rely on the optimizer.
        pass

    # If fallback needed, let's create a valid small packing
    centers = np.array([])
    radii = np.array([])
    
    # Let's try to fit 26 circles of radius 0.09 in a hexagonal pattern
    # 5 rows.
    # Rows with 5, 5, 5, 5, 6 circles?
    # Width for 6 circles of radius 0.09: 12*0.09 = 1.08 > 1.
    # So we cannot have 6 circles in a row with r=0.09.
    # So all rows must have <= 5 circles.
    # 5 rows * 5 circles = 25.
    # So we need 6 rows.
    # 6 rows of 5 circles = 30. We only need 26.
    # So we can have 4 rows of 5 and 2 rows of 3?
    # Or 5 rows of 5 and 1 row of 1?
    # 6 rows hexagonal height: (6-1)*sqrt(3)*r + 2r = (5*1.732 + 2)r = 10.66r.
    # If r=0.09, height = 0.96 <= 1. Fits!
    # So we can fit 6 rows of circles with r=0.09.
    # We need 26 circles.
    # Configuration: 5 rows of 5 circles, 1 row of 1 circle.
    # Wait, 5*5 + 1 = 26.
    # Let's place them.
    
    r = 0.09
    centers = []
    y = r
    for row in range(6):
        n_in_row = 5
        if row == 5:
            n_in_row = 1
        # Shift for hexagonal
        shift = r if row % 2 == 1 else 0
        # Actually shift should be such that distance is 2r.
        # Horizontal dist r, vertical sqrt(3)r => dist 2r.
        # So shift x by r.
        
        # For the single circle row, center it?
        if n_in_row == 1:
            centers.append([0.5, y])
        else:
            # 5 circles
            # Start x
            # Span of 5 circles: 5 * 2r = 10r = 0.9.
            # Available width 1. Margin 0.05 each side?
            # Centers at 0.05 + 0*r, 0.05 + 1*0.18? No.
            # Spacing 2r = 0.18.
            # x0 = (1 - 10r)/2 + r = 0.5 - 5r + r = 0.5 - 4r = 0.5 - 0.36 = 0.14.
            # Wait.
            # Width occupied by 5 circles is 10r.
            # Start at (1-10r)/2 = 0.05.
            # First center at 0.05 + r = 0.14? No.
            # Left edge at 0.05. Center at 0.05 + r = 0.14.
            # Last center at 0.05 + 9r = 0.95?
            # Let's just space them evenly.
            # x_coords = np.linspace(r, 1-r, 5)
            # But need to account for shift.
            
            # If shifted, we need to fit in [r, 1-r].
            # Shifted row starts at 2r?
            # If row 0 starts at r, row 1 starts at 2r?
            # Distance between (r, y0) and (2r, y1) is sqrt(r^2 + 3r^2) = 2r.
            # So yes, shift by r.
            # But if row 1 starts at 2r, last center for 5 circles:
            # 2r + 4*(2r) = 10r.
            # 10r = 0.9.
            # Right edge at 10r + r = 11r = 0.99 <= 1.
            # So shifted row of 5 circles fits!
            
            start_x = r if row % 2 == 0 else 2*r
            for k in range(n_in_row):
                cx = start_x + k * 2 * r
                centers.append([cx, y])
        
        y += math.sqrt(3) * r
        
    centers = np.array(centers)
    radii = np.array([r] * 26)
    
    # Verify
    if validate_packing(centers, radii):
        return centers, radii, np.sum(radii)
    else:
        # If validation fails, return a very safe packing
        # 26 circles of radius 0.05
        centers = np.random.rand(26, 2)
        radii = np.ones(26) * 0.05
        # Scale positions to be away from boundary
        centers = centers * 0.9 + 0.05
        # Re-validate
        if not validate_packing(centers, radii):
             # Force valid
             centers = np.array([[0.1 + i*0.3, 0.1 + j*0.3] for i in range(3) for j in range(3)]) # 9
             # ... this is getting messy.
             # The optimizer should have found a solution.
             pass

    # The optimizer result is preferred
    if best_params is not None:
        return best_params
    return centers, radii, np.sum(radii)

# Validation function provided in prompt
import numpy as np

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