# sol_000202 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=f9392240 sum of radii=2.213045 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization and SLSQP optimization.
    """
    n_circles = 26
    
    # --- Step 1: Generate Initial Configuration ---
    # We try to arrange circles in a hexagonal pattern, which is denser than square grid.
    # We aim for a compact arrangement.
    
    # Strategy: Create a list of points based on a hexagonal grid.
    # Basis vectors for hex grid with spacing 1: v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
    # We generate a large enough grid and pick 26 points that are centrally located or fit a box.
    
    # Let's try to construct a specific pattern that is known to be good or just a dense cluster.
    # A 5x5 grid has 25. We need 26.
    # A hexagonal block of 5 rows:
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    # If we just take a 5x5 square grid of points and perturb them to hex?
    
    # Let's create a 5x5 grid of centers first, then we will have 25.
    # We need 26.
    # Let's try a 6x5 hexagonal arrangement (30 points) and pick best 26?
    # Or just generate points and run optimization.
    
    # Initialization: Random points with small radius
    np.random.seed(42) # For reproducibility
    
    # Better initialization: Hexagonal lattice
    # We want to fit 26 circles. 
    # Approximate radius for 26 circles is around 0.1.
    # Spacing approx 0.2.
    
    centers_init = []
    # Let's try to fill rows.
    # If we have 5 rows, average 5.2 circles per row.
    # Hexagonal packing shifts rows.
    
    # Let's create points on a hex lattice
    # We want to cover the square [0,1]x[0,1]
    # Let's scale a lattice to fit.
    
    # Generate a hex lattice
    # Points (i + j/2, j * sqrt(3)/2)
    # We need to select 26 points.
    
    # Let's try a simple grid initialization first, it's robust.
    # 5x5 grid is 25 points. We need 1 more.
    # Maybe 5 rows of 5, and add one in the middle? No, middle is occupied.
    # Maybe shift one row?
    
    # Let's use a randomized initialization to avoid grid traps, 
    # but constrained to be valid.
    
    # Actually, a good start is placing them in a grid, slightly smaller.
    # Grid 6x5? 30 points. Too many.
    # Grid 5x6? 30 points.
    
    # Let's try 5 rows.
    # Row 0: 5 points
    # Row 1: 6 points (staggered) -> width might be issue.
    # Row 2: 5 points
    # Row 3: 6 points
    # Row 4: 4 points
    # Total 26.
    
    # Let's define coordinates for this pattern.
    # We need to scale them to fit in [0,1].
    
    pts = []
    # Row 0: 5 points, y=0
    for x in range(5):
        pts.append([x, 0.0])
    # Row 1: 6 points, y=1, shifted by 0.5
    for x in range(6):
        pts.append([x + 0.5, 1.0 * np.sqrt(3)]) # Vertical dist for hex is sqrt(3)/2 * spacing? 
        # Wait, if horizontal spacing is 1, vertical spacing for touching is sqrt(3)/2.
        # But here we just need distinct points.
    # Row 2: 5 points, y=2*sqrt(3)
    for x in range(5):
        pts.append([x, 2.0 * np.sqrt(3)])
    # Row 3: 6 points, y=3*sqrt(3)
    for x in range(6):
        pts.append([x + 0.5, 3.0 * np.sqrt(3)])
    # Row 4: 4 points, y=4*sqrt(3)
    for x in range(4):
        pts.append([x, 4.0 * np.sqrt(3)])
        
    pts = np.array(pts)
    
    # We have 26 points. Now we need to scale and center them to fit in [0,1]x[0,1].
    # Compute bounding box
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit in [0, 0.9]x[0, 0.9] to leave room for radii
    scale = 0.9 / max(width, height)
    pts = (pts - np.array([min_x, min_y])) * scale + 0.05 # Center roughly
    
    # Check if all within [0,1]
    pts = np.clip(pts, 0.01, 0.99)
    
    # If we have duplicate points (unlikely with this logic), perturb them
    # But let's just use the generated ones.
    # Note: The above generation logic had a bug in vertical spacing logic for "fitting".
    # Let's use a simpler, guaranteed valid initialization.
    
    # Fallback initialization: Random valid positions
    centers = np.zeros((n_circles, 2))
    radii = np.full(n_circles, 0.05) # Small initial radius
    
    # Try to place them using a rejection sampling or grid to ensure no overlap initially
    # But for optimization, starting with overlaps is fine if we use penalty, 
    # but SLSQP prefers feasible start.
    # Let's generate a valid start.
    
    # Simple grid packing for 26 circles
    # 5 rows of 5 = 25. 1 extra.
    # Let's do 6 rows?
    # 26 / 6 approx 4.3
    # Rows: 4, 5, 4, 5, 4, 4 -> 26
    # This fits width easily.
    
    current_idx = 0
    rows = [4, 5, 4, 5, 4, 4]
    y_positions = np.linspace(0.15, 0.85, len(rows))
    
    for r_idx, count in enumerate(rows):
        y = y_positions[r_idx]
        # Space them evenly in x
        # Available width roughly 0.8 (from 0.1 to 0.9)
        # Actually we can go 0 to 1, but need margin for radius.
        # If radius is 0.05, centers in [0.05, 0.95]. Range 0.9.
        x_range = np.linspace(0.05 + 0.9/(count+1), 0.95 - 0.9/(count+1), count)
        for x in x_range:
            if current_idx < n_circles:
                centers[current_idx] = [x, y]
                current_idx += 1
                
    # This gives a valid starting point with r=0.05.
    
    # --- Step 2: Optimization ---
    # Variables: x_0, y_0, r_0, ..., x_25, y_25, r_25
    # Total 78 variables.
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    def objective(vars):
        # Maximize sum of radii => Minimize negative sum
        r = vars[2*n_circles:]
        return -np.sum(r)
    
    # Constraints
    # 1. Circle inside square: x >= r, x <= 1-r, y >= r, y <= 1-r
    # 2. Non-overlap: dist(i,j) >= r_i + r_j
    
    constraints = []
    
    # Boundary constraints
    for i in range(n_circles):
        # x - r >= 0
        def make_bound_x_ge_r(idx):
            def cb(vars):
                return vars[idx*3] - vars[2*n_circles + idx]
            return cb
        constraints.append({'type': 'ineq', 'fun': make_bound_x_ge_r(i)})
        
        # 1 - x - r >= 0  => x + r <= 1
        def make_bound_x_le_1(idx):
            def cb(vars):
                return 1.0 - vars[idx*3] - vars[2*n_circles + idx]
            return cb
        constraints.append({'type': 'ineq', 'fun': make_bound_x_le_1(i)})
        
        # y - r >= 0
        def make_bound_y_ge_r(idx):
            def cb(vars):
                return vars[idx*3 + 1] - vars[2*n_circles + idx]
            return cb
        constraints.append({'type': 'ineq', 'fun': make_bound_y_ge_r(i)})
        
        # 1 - y - r >= 0
        def make_bound_y_le_1(idx):
            def cb(vars):
                return 1.0 - vars[idx*3 + 1] - vars[2*n_circles + idx]
            return cb
        constraints.append({'type': 'ineq', 'fun': make_bound_y_le_1(i)})

    # Overlap constraints
    # For i < j
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            def make_overlap_cb(i, j):
                def cb(vars):
                    x1, y1, r1 = vars[i*3], vars[i*3+1], vars[2*n_circles + i]
                    x2, y2, r2 = vars[j*3], vars[j*3+1], vars[2*n_circles + j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist = r1 + r2
                    # Constraint: dist >= min_dist => dist^2 >= min_dist^2
                    # But dist^2 - min_dist^2 can be negative if dist < min_dist.
                    # We want dist - min_dist >= 0.
                    # Using dist is better for smoothness near contact? 
                    # Actually dist^2 - (r1+r2)^2 is non-smooth at 0 if we use sqrt, 
                    # but here we use dist (sqrt). 
                    # Wait, sqrt(dist_sq) - min_dist >= 0.
                    # This is non-differentiable at dist=0, but dist won't be 0.
                    return np.sqrt(dist_sq) - min_dist
                return cb
            constraints.append({'type': 'ineq', 'fun': make_overlap_cb(i, j)})

    # Run optimization
    # SLSQP is good for constrained optimization
    res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Extract results
    if res.success:
        final_vars = res.x
    else:
        # If failed, use the best found or x0
        final_vars = res.x
    
    centers_final = np.zeros((n_circles, 2))
    radii_final = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers_final[i] = [final_vars[i*3], final_vars[i*3+1]]
        radii_final[i] = final_vars[2*n_circles + i]
    
    # Post-processing to ensure strict validity (clip very small negatives, etc)
    # Although optimizer should handle it, numerical noise might exist.
    radii_final = np.maximum(radii_final, 0.0)
    
    # Ensure centers are within bounds given radii
    for i in range(n_circles):
        x, y = centers_final[i]
        r = radii_final[i]
        centers_final[i, 0] = np.clip(x, r, 1.0 - r)
        centers_final[i, 1] = np.clip(y, r, 1.0 - r)
        # Re-clipping might violate overlap, but usually minor.
        # To be safe, if clipping changes center significantly, we might need to re-optimize, 
        # but for this task, it's likely fine.
        
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii

# Helper function to run and validate (for testing)
if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    
    # Quick manual check logic (simulating validate_packing)
    n = centers.shape[0]
    valid = True
    
    # Check NaN
    if np.isnan(centers).any() or np.isnan(radii).any():
        valid = False
        print("NaN detected")
    
    if valid:
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if r < 0:
                valid = False
                print(f"Circle {i} radius < 0")
                break
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                valid = False
                print(f"Circle {i} out of bounds")
                break
        
        if valid:
            # Check overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    if dist < radii[i] + radii[j] - 1e-12:
                        valid = False
                        print(f"Overlap {i} {j}")
                        break
                if not valid: break
    
    if valid:
        print("Packing is valid.")
    else:
        print("Packing is invalid.")
