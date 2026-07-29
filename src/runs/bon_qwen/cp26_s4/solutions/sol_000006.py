# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ca1ebfe6) state=e48bee11 sum of radii=2.557023 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We aim to place 26 circles. A hexagonal grid is dense.
    # We estimate a radius r to fit 26 circles. 
    # Area approx 1. 26 * pi * r^2 approx 0.9 (density).
    # r approx 0.105.
    
    # Let's generate points on a hex grid.
    # We will create a grid and pick the first 26 points that fit, 
    # or generate a grid scaled to fit n points.
    
    # Approximation: Square grid 5x5 has 25 points. 
    # We need 1 more. 
    # Let's try to fit a 5x6 hexagonal arrangement or similar.
    # Or just generate a grid and scale it.
    
    # Generate a triangular lattice
    points = []
    # Try to fit roughly in 1x1
    # Spacing dx = 1/5 = 0.2?
    # Let's try to generate more points than needed and select best, 
    # or just construct a specific layout.
    
    # Layout: 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 6 circles? Or 4?
    # Total 26.
    
    # Let's construct coordinates manually for a hex-like structure.
    # We will start with a loose grid and let optimization tighten it.
    
    # A 6x5 grid (30 points) might be a good template to pick 26 from?
    # Or just place 26 points in a hex pattern.
    
    # Let's try a simple grid first for robustness, then optimize.
    # 26 points. sqrt(26) approx 5.1. 
    # 5x6 grid?
    
    # Hexagonal coordinates generator
    # rows
    # Row i: y = i * sqrt(3)/2 * scale
    # x = j * scale + (i%2)*scale/2
    
    # Let's pick a scale such that we fit roughly 26.
    # If scale=0.2 (diameter 0.4? No, scale is distance between centers in x).
    # If scale = 0.2, 5 circles fit in width 1.
    # Vertical dist = 0.2 * sqrt(3)/2 approx 0.1732.
    # 5 rows height approx 4 * 0.1732 + radius_top + radius_bottom?
    # Actually center to center.
    
    # Let's just generate a dense set of candidate points and pick 26?
    # No, we need to optimize radii too.
    
    # Let's initialize centers in a hexagonal pattern with radius 0.1
    # This is feasible for 25 (5x5 grid). For 26, we might have slight overlap 
    # or need to shrink. The optimizer will handle it.
    
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.08 # Start small to ensure feasibility
    
    # Hexagonal packing coordinates
    # We want to fill the square.
    # Let's try to arrange in rows.
    # 5, 5, 5, 5, 4, 2 -> 26
    # Or 5, 6, 5, 6, 4?
    
    # Let's use a systematic approach:
    # Place points on a grid and prune or adjust.
    
    idx = 0
    # Try 6 rows
    # Row 0: 5
    # Row 1: 5 (shifted)
    # Row 2: 5
    # Row 3: 5 (shifted)
    # Row 4: 4 (shifted? no, aligned with 2?)
    # Row 5: 2
    
    # Let's just create a list of (x, y) for a hex grid and pick first 26
    candidates = []
    scale = 0.15 # generous spacing
    h = scale * np.sqrt(3) / 2
    
    # Generate grid points
    # y steps
    for row in range(10):
        y = 0.1 + row * h
        if y + 0.1 > 1.0: break # Rough check
        # x steps
        for col in range(10):
            x = 0.1 + col * scale + (row % 2) * (scale / 2)
            if x + 0.1 > 1.0: break
            candidates.append([x, y])
            
    # We have many candidates. We need to select 26 that are well distributed?
    # Actually, we can just pick the first 26 from a dense grid.
    # But better to pick a specific structure.
    
    # Let's try to fit 26 circles of radius 0.1.
    # 5x5 grid is 25.
    # Let's place 25 in 5x5 grid (centers 0.1, 0.3, 0.5, 0.7, 0.9)
    # And 1 extra in the center? (0.5, 0.5)
    # This might be a good start.
    
    grid_coords = []
    for i in range(5):
        for j in range(5):
            grid_coords.append([0.1 + i*0.2, 0.1 + j*0.2])
    
    # Add 1 extra at center
    grid_coords.append([0.5, 0.5])
    
    # This gives 26 points. 
    # However, center (0.5, 0.5) is already in the grid (i=2, j=2).
    # So we have duplicates.
    # Let's add at (0.5, 0.1) ? No.
    # Let's add at a hex position.
    # In a square grid, gaps are at (0.2, 0.2), etc.
    # But those are centers of holes.
    # Let's just use the 5x5 grid and perturb the 26th to a hole?
    # Or just use a hex grid directly.
    
    # Let's generate a proper hex grid with 26 points.
    # 5 rows of 5 is too many.
    # Rows: 5, 5, 5, 5, 4, 2?
    # Let's do:
    # Row 0: 5
    # Row 1: 5
    # Row 2: 5
    # Row 3: 5
    # Row 4: 4
    # Row 5: 2
    # Total 26.
    
    # Coordinates for hex packing with diameter d=0.2 (r=0.1)
    # Vertical spacing dy = d * sqrt(3)/2 approx 0.1732
    # Horizontal shift dx = d/2 = 0.1
    
    # Let's scale to fit.
    # Width required for 5 circles = 5*0.2 = 1.0. Fits exactly.
    # Height for 6 rows: 
    # 1st row center at y=r=0.1
    # Last row center at y = 0.1 + 5*0.1732 = 0.966
    # Top edge = 0.966 + 0.1 = 1.066 > 1.
    # So 6 rows of r=0.1 doesn't fit vertically.
    # We need smaller r or fewer rows.
    
    # Let's try 5 rows.
    # 5, 5, 5, 5, 6? No.
    # 5, 5, 5, 5, 4, 2 is 6 rows.
    # Maybe 5, 5, 5, 5, 5, 1?
    # Height for 6 rows is tight.
    
    # Let's just use a solver with a random-ish hex initialization.
    
    # Generate 26 points on a hex lattice, centered in square.
    # We can compute optimal r for this lattice later.
    
    # Let's create points
    pts = []
    # Try to fit in 1x1
    # Let's use a spacing of 0.18
    sp = 0.18
    pts_x = []
    pts_y = []
    
    # Fill grid
    for i in range(20):
        for j in range(20):
            x = i * sp + (j % 2) * (sp / 2)
            y = j * sp * np.sqrt(3) / 2
            pts_x.append(x)
            pts_y.append(y)
            
    pts_arr = np.array([pts_x, pts_y]).T
    
    # Translate and scale to fit in [0,1]
    min_x, max_x = np.min(pts_x), np.max(pts_x)
    min_y, max_y = np.min(pts_y), np.max(pts_y)
    
    # We want to pick 26 points that are spread out?
    # Actually, just pick the first 26 from a dense list might be clustered.
    # Better: pick points that form a shape.
    
    # Let's just use the 5x5 grid + 1 at (0.05, 0.05) corner?
    # Or just use a known good config.
    
    # Let's try a simple 5x5 grid (25 circles) + 1 circle in the middle of an edge?
    # Center of edge (0.5, 0.0)? No, radius 0.
    # (0.5, 0.1) is occupied.
    # (0.1, 0.5) occupied.
    # Holes are at (0.2, 0.2) etc? No, (0.1, 0.1) is center.
    # Hole is at (0.2, 0.2) relative to (0.1, 0.1)?
    # Distance from (0.1, 0.1) to (0.3, 0.3) is sqrt(0.08) ~ 0.28.
    # Hole at (0.2, 0.2) distance 0.14.
    
    # Let's just initialize with 26 random points in [0,1] and small radii.
    # But random is bad.
    # Let's use a grid of 6x6 = 36 points, and select 26?
    # No.
    
    # Let's go back to hex grid generation.
    # We will generate a hex grid with spacing 0.22 (r=0.11)
    # Then scale down to fit.
    
    # Actually, let's just define the centers explicitly for a 5x5 grid
    # and add one extra at a strategic location, e.g., replacing a circle
    # with two smaller ones? No, we need 26.
    
    # Let's use the 5x5 grid (25 circles) and add 1 circle at (0.5, 0.5) 
    # but we have to remove the one at (0.5, 0.5) first.
    # The 5x5 grid has a circle at (0.5, 0.5).
    # If we move the 26th circle to a hole, it must be small.
    # But we want to maximize sum.
    # Maybe we can expand the grid?
    # 26 circles. 
    # If we use a hexagonal packing, we can fit more.
    
    # Let's try to fit 26 circles in a hexagonal arrangement.
    # Rows: 5, 6, 5, 6, 4? Sum = 26.
    # Row widths:
    # 5 circles: width 5d.
    # 6 circles: width 6d.
    # If d=0.15, 6d=0.9 fits. 5d=0.75 fits.
    # Vertical spacing sqrt(3)/2 d ~ 0.13.
    # 5 rows height ~ 4*0.13 + 2r = 0.52 + 0.15 = 0.67. Fits easily.
    # So d=0.15 is very loose. We can increase d.
    # Max d limited by width of 6 circles? 6d <= 1 => d <= 0.166.
    # Max d limited by height?
    # With 5 rows, height constraint is loose.
    # So we can increase d until width hits 1.
    # d=0.166 => r=0.083. Sum = 26 * 0.083 = 2.15.
    # This is low.
    
    # What if we have fewer rows but more circles per row?
    # 3 rows of 9? 9d <= 1 => d <= 0.111 => r=0.055. Low.
    # We want d to be large.
    # Large d means fewer circles per row.
    # But we have fixed N=26.
    # So we need many rows.
    # Many rows => height constraint active.
    # Few rows => width constraint active.
    # Balance is around square shape.
    
    # 26 circles. sqrt(26) ~ 5.1.
    # So 5x5 grid is close.
    # For square grid, d=0.2, r=0.1. Sum=2.5.
    # For hex grid, we can pack denser.
    # But for fixed N, hex grid allows larger d?
    # Yes, hex packing density is higher.
    # For square grid, area fraction pi/4 = 0.785.
    # For hex grid, pi/sqrt(12) = 0.906.
    # So for same area, hex circles can be larger?
    # Area = N * pi * r^2.
    # Max area fraction ~ 0.906.
    # N * pi * r^2 <= 0.906 * 1.
    # r <= sqrt(0.906 / (26 * pi)) = sqrt(0.0111) = 0.105.
    # So theoretically r ~ 0.105 is possible.
    # Sum ~ 2.73.
    # The bottleneck is boundary.
    
    # Let's try to construct a hex grid with r=0.105.
    # d = 0.21.
    # 5 circles width 1.05 > 1.
    # So we cannot have 5 circles in a row if they are aligned?
    # In hex grid, rows are staggered.
    # Row 0: 5 circles. Width 5d? No, 4d + 2r?
    # If aligned to wall: centers r, 3r, 5r, 7r, 9r.
    # Right edge 9r+r = 10r = 5d.
    # So width is 5d.
    # If 5d > 1, cannot fit 5 aligned circles.
    # Can we fit 5 shifted circles?
    # Shifted centers 2r, 4r, ...
    # Right edge 10r+r = 11r = 5.5d. Worse.
    # So max circles in a row is floor(1/d).
    # If r=0.105, d=0.21. floor(1/0.21) = 4.
    # So max 4 circles per row.
    # To fit 26 circles, we need 26/4 = 6.5 rows -> 7 rows.
    # Height of 7 rows hex packing:
    # 6 gaps of height sqrt(3)/2 d ~ 0.866 * 0.21 = 0.182.
    # Total height 6 * 0.182 + 2r = 1.09 + 0.21 = 1.3.
    # Too tall.
    
    # So r=0.105 is impossible.
    # We need to find the optimal r.
    # Let's rely on the optimizer.
    
    # Initialization:
    # Let's create a grid of 6x5 points (30 points) and remove 4?
    # Or just 5x6 grid.
    # Let's place points on a 6x5 grid (spacing 0.2 in x, 0.2 in y? No 1/5=0.2).
    # 6 columns, 5 rows? 30 points.
    # x: 0.1, 0.3, 0.5, 0.7, 0.9, 1.1 (No)
    # 5 columns: 0.1, 0.3, 0.5, 0.7, 0.9.
    # 6 rows: 0.1, 0.3, 0.5, 0.7, 0.9, 1.1 (No).
    # Max 5x5.
    
    # Let's just use 5x5 grid (25 points) and add one at (0.05, 0.05) with small radius.
    # Optimizer will move them.
    
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # 5x5 grid
    k = 0
    for i in range(5):
        for j in range(5):
            centers[k] = [0.1 + i*0.2, 0.1 + j*0.2]
            k += 1
    
    # 26th circle
    centers[25] = [0.5, 0.5] # Overlaps, but optimizer handles it.
    # Actually 0.5, 0.5 is occupied by center (2,2).
    # Let's put it at corner?
    centers[25] = [0.05, 0.05]
    
    # Better initialization: Random perturbation of a valid packing?
    # A valid packing of 25 circles r=0.1 is 5x5.
    # We need 26.
    # Maybe 25 circles of r=0.095 and 1 of r=0.095?
    # 26 * 0.095 = 2.47.
    # Let's initialize all with r=0.09.
    
    radii[:] = 0.09
    
    # Perturb centers slightly to break symmetry?
    centers += np.random.normal(0, 0.001, centers.shape)
    
    # Optimization
    # Variables: x0, y0, r0, x1, y1, r1, ...
    x0 = centers.flatten()
    r0 = radii
    params0 = np.concatenate([x0, r0])
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)]) # x, y
        bounds.append((0.0, 0.5)) # r
        
    # Constraints
    # 1. Boundary
    # x - r >= 0 => x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    # 2. Non-overlap
    # dist >= r_i + r_j
    
    def constr_boundary(params):
        # Returns array of constraint values >= 0
        con = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            con.append(x - r)
            con.append(1.0 - x - r)
            con.append(y - r)
            con.append(1.0 - y - r)
        return np.array(con)

    def constr_overlap(params):
        con = []
        for i in range(n):
            for j in range(i+1, n):
                x_i, y_i = params[3*i], params[3*i+1]
                r_i = params[3*i+2]
                x_j, y_j = params[3*j], params[3*j+1]
                r_j = params[3*j+2]
                
                dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                con.append(dist - (r_i + r_j))
        return np.array(con)

    # Combine constraints
    # Using 'SLSQP', we can pass a list of dicts or a single function.
    # For many constraints, a single function returning array is better.
    
    def all_constraints(params):
        c1 = constr_boundary(params)
        c2 = constr_overlap(params)
        return np.concatenate([c1, c2])

    # To make it work with minimize, we need to specify nonlcons
    # nonlcons = {'type': 'ineq', 'fun': all_constraints}
    
    # However, SLSQP might struggle with 300+ constraints.
    # But let's try.
    
    # Objective
    def objective(params):
        return -np.sum(params[2::3]) # Sum of radii
    
    # Run optimization
    # Use a loop with multiple restarts?
    # Or just one run with good init.
    
    # The constraints are non-smooth at 0 distance? No, dist is smooth.
    # But gradient of dist is undefined at 0.
    # With r >= 0 and distinct centers, it's fine.
    
    # Let's run minimize
    result = minimize(objective, params0, method='SLSQP', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': all_constraints},
                      options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    
    # Extract results
    best_params = result.x
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    for i in range(n):
        best_centers[i, 0] = best_params[3*i]
        best_centers[i, 1] = best_params[3*i+1]
        best_radii[i] = best_params[3*i+2]
        
    # Validate
    # Just in case, clip radii to 0
    best_radii = np.maximum(best_radii, 0)
    
    return best_centers, best_radii, np.sum(best_radii)
