# sol_000199 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=39cc495f sum of radii=0.299918 correctness=1.0
# stdout(first 200): Optimizer error: cannot reshape array of size 26 into shape (26,2) Final sum of radii: 0.2999184868464684 Validation: True
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing():
    n = 26
    
    # --- Step 1: Initialization with Hexagonal Grid ---
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Try to fit 26 circles in a hexagonal pattern
    # We estimate an initial radius to place them
    # Area heuristic: n * pi * r^2 ~ 0.9 (density) * 1.0
    # r ~ sqrt(0.9 / (26 * pi)) ~ 0.105
    # But boundary effects reduce this. Let's start with r=0.08
    r_init = 0.08
    
    # Generate a large grid and pick 26 points that fit best?
    # Or just place them in rows.
    # Rows: 6, 5, 6, 5, 4 sums to 26.
    row_counts = [6, 5, 6, 5, 4]
    
    # Vertical spacing for hex packing with diameter 2r
    # dy = sqrt(3) * r
    # But we don't know r yet. Let's assume a packing density to estimate spacing.
    # If we want r=0.1, dy = 0.1732.
    # 5 rows height: 2r + 4*dy = 0.2 + 0.69 = 0.89. Fits.
    
    # Let's place centers based on a generic hex grid with spacing d=0.2 (r=0.1)
    # and then we will optimize.
    
    idx = 0
    y_curr = 0.1 # Start near bottom
    
    # We need to fit these in [0,1]x[0,1]
    # Let's just scatter them initially to avoid strict grid lock, 
    # but clustered in a hex shape.
    
    # Simple approach: Random initialization constrained to square
    # with some separation.
    np.random.seed(42)
    
    # Better initialization: 
    # Place in a grid, then shift to hex.
    # Grid 6x5 = 30 points. Remove 4.
    # Grid points
    pts = []
    # Try 6 columns, 5 rows
    # Spacing 1.0/7 for x, 1.0/6 for y ?
    # Let's use a denser grid to ensure 26 fit
    for i in range(7): # 0..6
        for j in range(6): # 0..5
            x = 0.1 + i * 0.13 # spread out
            y = 0.1 + j * 0.15
            if x < 0.9 and y < 0.9:
                pts.append([x, y])
    
    # If we have more than 26, pick the first 26
    # If fewer, add random points
    while len(pts) < n:
        pts.append([np.random.rand(), np.random.rand()])
        
    centers = np.array(pts[:n])
    radii = np.ones(n) * 0.05 # Small initial radius
    
    # --- Step 2: Iterative Expansion and Relaxation ---
    # We will try to grow radii and resolve conflicts
    
    max_iterations = 500
    growth_rate = 1.01 # 1% growth per step
    repulsion_strength = 0.1
    
    for step in range(max_iterations):
        # Try to grow radii
        radii *= growth_rate
        
        # Check constraints and resolve
        valid = True
        # Boundary check
        for i in range(n):
            cx, cy = centers[i]
            r = radii[i]
            # Push center inward if touching boundary
            if cx - r < 0: centers[i, 0] += (r - cx)
            elif cx + r > 1: centers[i, 0] -= (cx + r - 1)
            
            if cy - r < 0: centers[i, 1] += (r - cy)
            elif cy + r > 1: centers[i, 1] -= (cy + r - 1)
            
            # Clamp radii to max possible given current position
            max_r_x = min(centers[i, 0], 1 - centers[i, 0])
            max_r_y = min(centers[i, 1], 1 - centers[i, 1])
            max_r = min(max_r_x, max_r_y)
            radii[i] = min(radii[i], max_r)

        # Overlap check and repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap! Push apart
                    overlap = min_dist - dist
                    direction = dist_vec / dist
                    
                    # Move both apart proportional to radius (or equally)
                    # Move i by +overlap/2, j by -overlap/2
                    move = direction * (overlap / 2.0)
                    centers[i] += move
                    centers[j] -= move
                    
                    # Clamp to square after move
                    centers[i] = np.clip(centers[i], [0,0], [1,1])
                    centers[j] = np.clip(centers[j], [0,0], [1,1])

    # --- Step 3: Refinement with SciPy Optimizer ---
    # Maximize sum(radii)
    # Variables: flattened [x1, y1, r1, x2, y2, r2, ...]
    # But r is coupled with x,y.
    # Actually, we can just optimize x, y and compute max r?
    # No, max r depends on all neighbors.
    # Let's optimize full vector.
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Objective: Negative sum of radii (to minimize)
    def objective(vars):
        # vars: x1, y1, r1, ...
        # r is at indices 2, 5, 8, ...
        r_sum = 0
        for i in range(n):
            r_sum += vars[3*i + 2]
        return -r_sum

    # Constraints
    # 1. r_i >= 0 (implicit if we bound, but let's enforce)
    # 2. x_i - r_i >= 0
    # 3. x_i + r_i <= 1
    # 4. y_i - r_i >= 0
    # 5. y_i + r_i <= 1
    # 6. dist_ij >= r_i + r_j
    
    # SLSQP allows inequality constraints g(x) >= 0
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        xi = 3*i
        yi = 3*i + 1
        ri = 3*i + 2
        
        # x - r >= 0
        def make_c_x1(i):
            return lambda vars: vars[3*i] - vars[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': make_c_x1(i)})
        
        # 1 - (x + r) >= 0 => x + r <= 1
        def make_c_x2(i):
            return lambda vars: 1.0 - (vars[3*i] + vars[3*i + 2])
        constraints.append({'type': 'ineq', 'fun': make_c_x2(i)})
        
        # y - r >= 0
        def make_c_y1(i):
            return lambda vars: vars[3*i + 1] - vars[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': make_c_y1(i)})
        
        # 1 - (y + r) >= 0
        def make_c_y2(i):
            return lambda vars: 1.0 - (vars[3*i + 1] + vars[3*i + 2])
        constraints.append({'type': 'ineq', 'fun': make_c_y2(i)})
            
        # r >= 0
        def make_c_r(i):
            return lambda vars: vars[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': make_c_r(i)})

    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    # dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            
            def make_c_overlap(i, j):
                return lambda vars: \
                    (vars[xi] - vars[xj])**2 + (vars[yi] - vars[yj])**2 - \
                    (vars[ri] + vars[rj])**2
            constraints.append({'type': 'ineq', 'fun': make_c_overlap(i, j)})

    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Run optimizer
    # SLSQP might struggle with 78 vars and 300+ constraints.
    # But let's try. We can limit options.
    try:
        result = scipy.optimize.minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        if result.success or result.fun < -1.0: # Just check if it did something
            x_opt = result.x
            centers_opt = x_opt[0::3].reshape(n, 2)
            radii_opt = x_opt[2::3]
            # Check validity
            if validate_packing(centers_opt, radii_opt):
                centers = centers_opt
                radii = radii_opt
                print(f"Optimizer improved sum to {-result.fun}")
            else:
                print("Optimizer result invalid, keeping heuristic.")
        else:
            print(f"Optimizer failed or no improvement. Success: {result.success}, Fun: {result.fun}")
    except Exception as e:
        print(f"Optimizer error: {e}")

    # Final validation and clipping
    # Ensure strict validity
    for i in range(n):
        centers[i] = np.clip(centers[i], [0,0], [1,1])
        radii[i] = max(0, radii[i])
        # Re-clamp r based on center
        r = radii[i]
        cx, cy = centers[i]
        if cx - r < 0: r = cx
        if cx + r > 1: r = 1 - cx
        if cy - r < 0: r = cy
        if cy + r > 1: r = 1 - cy
        radii[i] = r

    # Final check for overlaps and fix small ones
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - 1e-12:
                # Resolve by shrinking both
                overlap = (radii[i] + radii[j]) - dist
                radii[i] -= overlap / 2
                radii[j] -= overlap / 2
                # Ensure non-negative
                radii[i] = max(0, radii[i])
                radii[j] = max(0, radii[j])

    total_radius = np.sum(radii)
    print(f"Final sum of radii: {total_radius}")
    print(f"Validation: {validate_packing(centers, radii)}")
    
    return centers, radii, total_radius

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
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

# Run the packing
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
