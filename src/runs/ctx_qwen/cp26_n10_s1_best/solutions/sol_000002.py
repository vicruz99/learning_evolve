# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d5d6e849) state=f611bc76 sum of radii=2.453410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_hexagonal_initialization(n):
    """
    Generates an initial configuration of n points in a hexagonal lattice 
    inside a unit square, scaled to be reasonably compact.
    """
    centers = []
    # Hexagonal grid parameters
    # We try to fit points in a roughly square aspect ratio
    # Hexagonal spacing: horizontal 1.5, vertical sqrt(3)/2 (for diameter 1)
    # But let's just place them in rows
    
    # Estimate grid size
    cols = int(np.ceil(np.sqrt(n * 2.0))) # slightly more columns
    rows = int(np.ceil(n / cols))
    
    # Adjust to fit in square better
    # A rough square of points
    side = int(np.ceil(np.sqrt(n)))
    
    current_n = 0
    r_est = 1.0 / (2.0 * side) # rough radius estimate for square grid
    
    y_step = np.sqrt(3) * 1.0 # normalized vertical step relative to horizontal
    # Let's normalize to unit square size roughly
    # We will optimize positions later, this is just a seed.
    
    # Simple hexagonal packing generator
    # Row 0: 0, 2, 4...
    # Row 1: 1, 3, 5...
    # spacing 1 (diameter)
    
    # Let's create a grid and pick n points
    # We want a shape that fits in 1x1
    
    pts = []
    # Try a bounding box of roughly 5x5 or 6x6
    # Let's try to fill rows
    
    # Approximate number of circles per row for hex packing in 1x1
    # If r is small, we can fit many.
    # Let's just generate a grid of potential centers and pick the first n
    # that fit well, or just a rectangular block.
    
    # Let's try a 6x5 block (30 points) and remove 4
    # Or 5x6
    
    # Coordinates for hex grid with spacing 1 (diameter)
    # x = col * 1.5 + (row % 2) * 0.75
    # y = row * 0.866
    
    # We need to scale this to fit in [0,1]x[0,1]
    # Let's generate a 6x6 grid and normalize
    
    grid_pts = []
    for r in range(6):
        for c in range(6):
            x = c * 1.5 + (r % 2) * 0.75
            y = r * 0.8660254
            grid_pts.append([x, y])
            
    # We have 36 points. We need 26.
    # Let's select the ones that form a compact shape.
    # Actually, just picking the first 26 from a dense grid might leave gaps at edges.
    # Better to center them.
    
    # Let's try a specific arrangement for 26
    # 5, 6, 5, 6, 4 rows?
    # Or just use the optimizer to find it from a good start.
    
    # Let's create a square-ish block of hex points
    # 5 rows, 5-6 points each
    
    selected_pts = []
    count = 0
    # Try to pick points close to center
    # Center of grid roughly
    cx, cy = 4.5, 3.5 # approx center of 6x6 grid
    
    # Calculate distances to center and sort
    dists = []
    for i, p in enumerate(grid_pts):
        d = (p[0] - cx)**2 + (p[1] - cy)**2
        dists.append((d, i))
    
    dists.sort()
    
    for d, idx in dists:
        selected_pts.append(grid_pts[idx])
        count += 1
        if count == n:
            break
            
    # Normalize to unit square
    xs = [p[0] for p in selected_pts]
    ys = [p[1] for p in selected_pts]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit inside [0,1] with some padding
    scale = 0.9 / max(width, height)
    
    final_centers = np.zeros((n, 2))
    for i, p in enumerate(selected_pts):
        final_centers[i, 0] = (p[0] - min_x) * scale + (1.0 - width * scale) / 2.0
        final_centers[i, 1] = (p[1] - min_y) * scale + (1.0 - height * scale) / 2.0
        
    return final_centers

def objective(vars, n):
    """
    Objective function to maximize radius r (minimize -r).
    vars = [x1, y1, ..., xn, yn, r]
    """
    return -vars[-1]

def constraint_boundary(vars, n):
    """
    Constraints: r <= xi <= 1-r, r <= yi <= 1-r
    """
    r = vars[-1]
    constraints = []
    for i in range(n):
        x = vars[2*i]
        y = vars[2*i+1]
        # x - r >= 0 => x - r
        constraints.append(x - r)
        # 1 - x - r >= 0 => 1 - x - r
        constraints.append(1 - x - r)
        # y - r >= 0
        constraints.append(y - r)
        # 1 - y - r >= 0
        constraints.append(1 - y - r)
    return np.array(constraints)

def constraint_distance(vars, n):
    """
    Constraints: dist(i, j) >= 2r
    """
    r = vars[-1]
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = vars[2*i], vars[2*i+1]
            x2, y2 = vars[2*j], vars[2*j+1]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            # dist >= 2r => dist^2 >= 4r^2
            constraints.append(dist_sq - 4 * r**2)
    return np.array(constraints)

def run_packing():
    n = 26
    
    # 1. Initial guess
    # Start with a hexagonal packing
    centers_init = get_hexagonal_initialization(n)
    
    # Initial radius guess. 
    # For 26 circles, equal radius approx 0.1
    r_init = 0.10
    
    # Flatten variables
    x0 = np.hstack([centers_init.flatten(), [r_init]])
    
    # 2. Optimization
    # We use SLSQP to maximize r (minimize -r)
    # Constraints are inequalities >= 0
    
    def get_constraints(vars, n):
        return np.hstack([constraint_boundary(vars, n), constraint_distance(vars, n)])

    # Setup bounds: x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
    bounds.append((0, 0.5))   # r
    
    # Optimization options
    options = {'maxiter': 1000, 'ftol': 1e-9}
    
    # Run optimizer
    # Note: scipy minimize with many constraints can be slow or unstable.
    # We might need to run it a few times or use a robust method.
    # Trust-constr is often better for constraints but slower.
    # SLSQP is good balance.
    
    try:
        result = minimize(objective, x0, args=(n,), method='SLSQP', 
                          bounds=bounds, 
                          constraints={'type': 'ineq', 'fun': lambda v: get_constraints(v, n)},
                          options=options)
        
        if result.success:
            centers_opt = result.x[:2*n].reshape((n, 2))
            r_opt = result.x[2*n]
        else:
            # If optimization fails, fallback to heuristic or just return init
            # But SLSQP usually works if initial guess is feasible.
            # Check if initial guess is feasible
            # If not, we might need to reduce r_init
            centers_opt = centers_init
            r_opt = r_init
            
    except Exception as e:
        # Fallback
        centers_opt = centers_init
        r_opt = r_init

    # 3. Refine: The optimizer maximizes r. 
    # But the constraints are strict. 
    # If we found a solution, r_opt is the max equal radius.
    # We need to ensure the solution is valid (numerical tolerance).
    # The optimizer might push r slightly beyond feasible due to numerical noise.
    # We can clamp r or slightly reduce it.
    
    # Let's verify and adjust if needed
    # Check overlaps
    valid = True
    min_dist = 1.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers_opt[i] - centers_opt[j])
            if d < 2 * r_opt - 1e-7:
                # Overlap detected, reduce r
                r_opt = d / 2.0 - 1e-9
                valid = False # Mark to check others
    
    # Check boundaries
    for i in range(n):
        x, y = centers_opt[i]
        r_min = min(x, 1-x, y, 1-y)
        if r_min < r_opt - 1e-7:
            r_opt = r_min - 1e-9

    # If we reduced r, it's safer.
    
    # 4. Check if unequal radii can improve sum.
    # With equal radii, sum = 26 * r_opt.
    # The target is 2.636.
    # If 26 * r_opt < 2.636, we might try unequal.
    # But for 26 circles, equal is likely near optimal.
    # Let's check the result.
    
    # To be safe, we can try a quick local optimization on radii if sum is low?
    # But that's complex. Let's rely on the equal radius packing.
    # If the optimizer works well, r should be around 0.102 or higher.
    # 26 * 0.102 = 2.652 > 2.636.
    
    # Just in case, let's make sure we don't return a packing that is invalid due to precision.
    # We can slightly shrink radii to be safe.
    r_final = r_opt * 0.9999
    
    centers = centers_opt
    radii = np.full(n, r_final)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
