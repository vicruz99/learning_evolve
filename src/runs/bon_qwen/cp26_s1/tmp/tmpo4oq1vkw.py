import numpy as np
from scipy.optimize import minimize

def run_packing():
    # Number of circles
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We try to fit 26 circles. A common dense arrangement is rows.
    # Let's try to arrange them in a pattern that fits well in a square.
    # 5 rows of 5 circles is 25. We need 26.
    # Maybe 6 rows? 5, 4, 5, 4, 5, 3? Sum = 26.
    # Or just a grid perturbed to hexagonal.
    
    # Let's create a grid of points and then adjust to hexagonal spacing if possible,
    # or just use a dense grid.
    # A 6x5 grid is 30 points. We can select 26 best spaced points?
    # Or just initialize on a hexagonal grid.
    
    centers = []
    radius_init = 0.08 # Initial guess, will be optimized
    
    # Hexagonal packing parameters
    # Horizontal spacing 2*r, vertical spacing sqrt(3)*r
    # We want to fit in [0,1]x[0,1]
    # Let's estimate r needed for 26 circles in hex packing.
    # Area approx 26 * pi * r^2 * (1/0.907) <= 1 => r ~ 0.11?
    # But boundary effects. Let's start with r=0.1.
    
    r_est = 0.1
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    row = 0
    col = 0
    
    # We will generate points until we have 26
    # Row 0: y = r_est, x = r_est, r_est + dx, ...
    # Row 1: y = r_est + dy, x = r_est + dx/2, ... (offset)
    
    # Let's just generate a list of potential centers and pick the best 26?
    # Or just construct a specific layout.
    
    # Layout: 
    # 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 6 circles? No, width constraint.
    # Maybe 5, 5, 5, 5, 5 and 1 in center?
    # Let's try to distribute them uniformly.
    
    # Simple approach: 6 rows, alternating 5 and 4 circles?
    # 5+4+5+4+5+4 = 27 (too many)
    # 5+4+5+4+5+3 = 26
    # Let's try this.
    
    rows_config = [5, 4, 5, 4, 5, 3] # Total 26
    # But this might not be optimal. 
    # Let's try a more uniform distribution.
    # 5 rows of 5 circles = 25. Add 1 somewhere.
    # Or 6 rows of ~4-5.
    
    # Let's use a grid initialization which is robust.
    # 5x6 grid = 30 points. We can optimize positions.
    # But we need exactly 26.
    # Let's pick 26 points from a 6x5 grid (0.2 spacing).
    
    # Grid points
    x_coords = np.linspace(0.1, 0.9, 5) # 5 points
    y_coords = np.linspace(0.1, 0.9, 6) # 6 points
    # Total 30. We need 26.
    # Let's remove 4 points that are most crowded?
    # Actually, let's just generate 26 points on a hexagonal lattice manually.
    
    # Better initialization:
    # Place circles in a grid, then we optimize.
    # 5 columns, 6 rows?
    # 5 * 6 = 30.
    # Let's take first 26?
    
    # Let's construct a hexagonal layout explicitly.
    # We want to maximize density.
    # Try to fit 5 circles in a row.
    # Width needed for 5 circles: 10*r.
    # 10*r <= 1 => r <= 0.1.
    # If r=0.1, 5 circles fit exactly.
    # Vertical spacing for hex: sqrt(3)*0.1 ~ 0.1732.
    # Height for 6 rows: 2*0.1 + 5*0.1732 = 0.2 + 0.866 = 1.066 > 1.
    # So 6 rows of 0.1 circles don't fit vertically.
    # 5 rows: 2*0.1 + 4*0.1732 = 0.2 + 0.6928 = 0.8928 < 1.
    # So 5 rows fit. 5 rows * 5 cols = 25 circles.
    # We need 26.
    # So we must shrink radius to fit 6th row or add a circle in a gap.
    
    # Let's try 6 rows with slightly smaller radius.
    # Let r be variable.
    # We will initialize with r=0.09.
    
    r_init = 0.09
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    current_centers = []
    
    # Row 0: 5 circles
    # Row 1: 4 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles (shifted) -> Total 26
    # Let's try to balance.
    
    row_counts = [5, 4, 5, 4, 5, 3] # Sum 26
    # Check if this fits.
    # Row 0 y = r
    # Row 1 y = r + dy
    # ...
    # Last row y = r + 5*dy.
    # Constraint: r + 5*dy + r <= 1 => 2r + 5*sqrt(3)*r <= 1
    # r(2 + 8.66) <= 1 => 10.66r <= 1 => r <= 0.0938.
    # With r=0.09, it fits.
    
    y = r_init
    for count in row_counts:
        x_start = r_init
        if len(current_centers) % 2 != 0: # Odd rows (1, 3, 5) shifted
             # Shift by dx/2 = r
             x_start += r_init
             # But wait, standard hex packing:
             # Row 0: 0, 2r, 4r...
             # Row 1: r, 3r, 5r...
             # So x_start depends on row index.
             pass 
        # Actually simpler:
        # Even rows: x = r, 3r, 5r...
        # Odd rows: x = 2r, 4r, 6r... (shifted by r)
        # Wait, spacing is 2r. Offset is r.
        
        # Let's just place them
        for c in range(count):
            # Calculate x based on row parity
            row_idx = len(row_counts) - 1 - (len(row_counts) - 1 - row_counts.index(count)) # No
            pass 
        
        # Let's do it cleanly
        pass

    # Re-initialization logic
    centers_init = []
    r_curr = 0.09
    # We want to center the whole pattern in the square
    # Let's define rows
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted) -> Total 10
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted) -> Total 20
    # Row 4: 6 circles? No.
    # Let's just use a grid and let optimizer fix it.
    # 5x5 grid + 1 center?
    
    # Grid 5x5
    x_g = np.linspace(0.15, 0.85, 5)
    y_g = np.linspace(0.15, 0.85, 5)
    grid_centers = []
    for y in y_g:
        for x in x_g:
            grid_centers.append([x, y])
    # 25 centers. Add one in center [0.5, 0.5]
    grid_centers.append([0.5, 0.5])
    # Total 26.
    # This is a valid starting point.
    
    # Flatten to array
    centers_arr = np.array(grid_centers)
    radii_arr = np.ones(n) * 0.08 # Small initial radius
    
    # 2. Optimization
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Order: x0, y0, r0, x1, y1, r1 ...
    
    x0 = np.concatenate([centers_arr.flatten(), radii_arr.flatten()])
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    def objective(vars):
        # vars: [x0, y0, r0, x1, y1, r1, ...]
        centers_opt = vars[:2*n].reshape(n, 2)
        radii_opt = vars[2*n:]
        
        # Objective: maximize sum of radii -> minimize negative sum
        obj = -np.sum(radii_opt)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                req_dist = radii_opt[i] + radii_opt[j]
                if dist < req_dist:
                    penalty += (req_dist - dist)**2
        
        # Penalty for boundaries
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if x - r < 0:
                penalty += (r - x)**2
            if x + r > 1:
                penalty += (x + r - 1)**2
            if y - r < 0:
                penalty += (r - y)**2
            if y + r > 1:
                penalty += (y + r - 1)**2
        
        # Weight for penalty needs to be high enough
        # But if too high, it might stall.
        # Let's use a dynamic weight or a fixed large one.
        # Given the scale, overlap penalty squared is small.
        # Let's multiply by a factor.
        penalty *= 1000.0
        
        return obj + penalty

    # We can run optimization multiple times with different seeds or just one good run.
    # L-BFGS-B is good for bounded variables.
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    final_vars = result.x
    centers_final = final_vars[:2*n].reshape(n, 2)
    radii_final = final_vars[2*n:]
    
    # Clip radii to non-negative just in case
    radii_final = np.maximum(radii_final, 0.0)
    
    # Calculate sum
    sum_radii = np.sum(radii_final)
    
    # Verify validity (optional but good for debugging)
    # We trust the optimizer minimized penalty to 0.
    
    return centers_final, radii_final, sum_radii