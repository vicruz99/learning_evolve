# sol_000111 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=4f5c20af sum of radii=2.470358 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution
import sys
import io

# --- Validation Function (Provided) ---
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

# --- Helper Functions ---

def generate_hex_grid(n, scale=1.0):
    """Generate n points in a hexagonal lattice pattern centered at origin."""
    # Approximate spacing for hex grid
    # We want to generate slightly more than n points and then select the n closest to center
    # or just generate a block and pick.
    
    # Let's generate a grid of points
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3)))) + 2
    cols = int(np.ceil(np.sqrt(n))) + 2
    
    points = []
    for i in range(-rows, rows):
        for j in range(-cols, cols):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * (np.sqrt(3) / 2)
            points.append([x, y])
    
    points = np.array(points)
    
    # Center the points
    center = np.mean(points, axis=0)
    points -= center
    
    # Sort by distance to origin to pick the 'n' most central ones
    dists = np.sqrt(np.sum(points**2, axis=1))
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices[:n]]
    
    return points * scale

def get_overlap_penalty(centers, radii):
    """Calculate total overlap penalty."""
    n = centers.shape[0]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return penalty

def get_boundary_penalty(centers, radii):
    """Calculate penalty for being outside [0,1]x[0,1]."""
    penalty = 0.0
    for i in range(len(radii)):
        x, y = centers[i]
        r = radii[i]
        # Left
        if x - r < 0: penalty += (x - r) ** 2
        # Right
        if x + r > 1: penalty += (x + r - 1) ** 2
        # Bottom
        if y - r < 0: penalty += (y - r) ** 2
        # Top
        if y + r > 1: penalty += (y + r - 1) ** 2
    return penalty

def objective_function_global(params, centers_template, n):
    """
    Objective for Differential Evolution.
    params: [scale, dx, dy]
    """
    scale, dx, dy = params
    # Scale the template and shift
    # Ensure scale is positive
    if scale <= 1e-9: scale = 1e-9
    
    current_centers = centers_template * scale + np.array([dx, dy])
    
    # We assume equal radii for the global search to find the optimal "size"
    # Radius is half the spacing scale? 
    # In our template, points are spaced by 1.0 horizontally (roughly).
    # If scale is s, spacing is s. Radius should be s/2.
    # But let's just use a fixed radius based on scale for the search, 
    # and let the local optimizer handle variable radii later.
    # Actually, simpler: just return -scale (maximize scale).
    # But we must check validity. If invalid, return huge value.
    
    r = scale / 2.0
    radii = np.full(n, r)
    
    # Check constraints
    # Boundary
    valid = True
    for i in range(n):
        x, y = current_centers[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            valid = False
            break
    
    if not valid:
        return 1000.0 # Penalty
        
    # Overlap
    overlap = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((current_centers[i] - current_centers[j]) ** 2))
            if dist < 2*r - 1e-6:
                return 1000.0 # Penalty
                
    # Maximize scale -> Minimize -scale
    return -scale

def objective_function_local(params, n):
    """
    Objective for local refinement (minimize negative sum of radii + penalties).
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        idx = 3 * i
        centers[i] = params[idx:idx+2]
        radii[i] = params[idx+2]
        
    # Objective: Maximize sum(radii) => Minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Penalty for overlaps
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # Quadratic penalty
                overlap_pen += 100 * (min_dist - dist) ** 2
                
    # Penalty for boundaries
    bound_pen = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: bound_pen += 100 * (x - r)**2
        if x + r > 1: bound_pen += 100 * (x + r - 1)**2
        if y - r < 0: bound_pen += 100 * (y - r)**2
        if y + r > 1: bound_pen += 100 * (y + r - 1)**2
        
    # Penalty for negative radii (though bounds handle it)
    neg_pen = 0.0
    for r in radii:
        if r < 0: neg_pen += 100 * r**2
        
    return obj + overlap_pen + bound_pen + neg_pen

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Generate Hexagonal Template
    # We generate a template with unit spacing.
    # In a hex grid, horizontal dist is 1, vertical is sqrt(3)/2.
    # We will scale this template.
    centers_template = generate_hex_grid(n, scale=1.0)
    
    # 2. Global Optimization to find best scale and position
    # Bounds for scale: [0.01, 1.0], dx: [-0.2, 0.2], dy: [-0.2, 0.2]
    # Actually dx, dy can be larger, but we want to center it.
    # Let's allow dx, dy in [-1, 1] just in case, but start near 0.
    bounds_global = [(0.05, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    
    # Differential Evolution might take time, but for 3 variables it's fast.
    # We run it to find the largest uniform packing.
    # Note: This finds equal radii solution.
    
    # To speed up, we can just try a few scales or use minimize directly?
    # Let's use DE for robustness.
    # However, DE is stochastic. We can run it once.
    
    # A simpler global approach: 
    # The optimal equal-radius packing for 26 circles is likely very close to the hex grid scaling.
    # We can just use `minimize` with a good initial guess derived from geometry.
    
    # Initial guess:
    # Bounding box of template.
    x_min, x_max = np.min(centers_template[:, 0]), np.max(centers_template[:, 0])
    y_min, y_max = np.min(centers_template[:, 1]), np.max(centers_template[:, 1])
    width_t = x_max - x_min
    height_t = y_max - y_min
    
    # To fit in 1x1 with margin 2r on each side (diameter), 
    # effectively we need to fit the shape of diameter D_template into 1-2r?
    # Actually, if we scale by s, radius is s/2.
    # The shape spans width_t * s.
    # We need width_t * s + 2*(s/2) <= 1 ? No, centers must be in [r, 1-r].
    # So width_t * s <= 1 - 2r = 1 - s.
    # s * (width_t + 1) <= 1 => s <= 1 / (width_t + 1).
    # Similarly for y.
    
    s_x = 1.0 / (width_t + 1.0)
    s_y = 1.0 / (height_t + 1.0)
    s_init = min(s_x, s_y)
    
    # Initial radius
    r_init = s_init / 2.0
    
    # Initial centers
    centers_init = centers_template * s_init
    # Shift to center
    cx = np.mean(centers_init[:, 0])
    cy = np.mean(centers_init[:, 1])
    centers_init[:, 0] += 0.5 - cx
    centers_init[:, 1] += 0.5 - cy
    
    # 3. Local Optimization (Variable Radii)
    # We optimize centers and radii simultaneously to maximize sum.
    
    # Initial params for local optimization
    # x1, y1, r1, x2, y2, r2 ...
    initial_params = np.zeros(3 * n)
    for i in range(n):
        initial_params[3*i] = centers_init[i, 0]
        initial_params[3*i+1] = centers_init[i, 1]
        initial_params[3*i+2] = r_init # Start with equal radii
        
    # Bounds for local optimization
    # x, y in [0, 1], r in [0, 0.5]
    lb = [0.0] * (2 * n) + [0.0] * n # x, y >= 0, r >= 0
    ub = [1.0] * (2 * n) + [0.5] * n # x, y <= 1, r <= 0.5
    bounds_local = list(zip(lb, ub))
    
    # Run minimization
    # SLSQP is good for constrained, but we used penalty method here, so L-BFGS-B or BFGS is fine.
    # L-BFGS-B respects bounds.
    res = minimize(objective_function_local, initial_params, args=(n,), method='L-BFGS-B', bounds=bounds_local, 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    best_params = res.x
    
    # Extract solution
    centers_sol = np.zeros((n, 2))
    radii_sol = np.zeros(n)
    for i in range(n):
        centers_sol[i] = best_params[3*i : 3*i+2]
        radii_sol[i] = best_params[3*i+2]
        
    # 4. Verification and Cleaning
    # Ensure radii are non-negative (bounds should handle, but just in case)
    radii_sol = np.maximum(radii_sol, 0.0)
    
    # Clamp centers to [0,1] just in case
    centers_sol[:, 0] = np.clip(centers_sol[:, 0], 0.0, 1.0)
    centers_sol[:, 1] = np.clip(centers_sol[:, 1], 0.0, 1.0)
    
    # If any circle is too large for its position (sticking out), we might need to shrink it.
    # But the penalty function should have handled it.
    # However, L-BFGS-B might stop before perfect validity if penalty gradient is weak.
    # Let's do a correction pass.
    
    # Correction: ensure boundaries
    for i in range(n):
        x, y = centers_sol[i]
        r = radii_sol[i]
        max_r = min(x, 1-x, y, 1-y)
        if max_r < 1e-9:
            # Circle is outside or on edge with no room. Move to center? 
            # Or just set r=0.
            radii_sol[i] = 0.0
        else:
            if r > max_r + 1e-9:
                radii_sol[i] = max_r
                
    # Correction: resolve overlaps
    # Simple iterative shrink/shift
    for _ in range(50): # A few passes
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_sol[i, 0] - centers_sol[j, 0]
                dy = centers_sol[i, 1] - centers_sol[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii_sol[i] + radii_sol[j]
                
                if dist < sum_r - 1e-12 and dist > 1e-12:
                    # Overlap
                    # Push apart
                    overlap = sum_r - dist
                    # Push centers apart by overlap/2
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Move i away from j
                    centers_sol[i, 0] += nx * (overlap/2 + 1e-6)
                    centers_sol[i, 1] += ny * (overlap/2 + 1e-6)
                    
                    # Move j away from i
                    centers_sol[j, 0] -= nx * (overlap/2 + 1e-6)
                    centers_sol[j, 1] -= ny * (overlap/2 + 1e-6)
                    
                    # Clamp centers
                    centers_sol[:, 0] = np.clip(centers_sol[:, 0], 0.0, 1.0)
                    centers_sol[:, 1] = np.clip(centers_sol[:, 1], 0.0, 1.0)
                    
                    # Re-check boundary radii
                    for k in range(n):
                        x, y = centers_sol[k]
                        r = radii_sol[k]
                        max_r = min(x, 1-x, y, 1-y)
                        if r > max_r:
                            radii_sol[k] = max_r

    # Final validation
    is_valid = validate_packing(centers_sol, radii_sol)
    
    if not is_valid:
        # Fallback to a known valid packing (e.g. small equal circles)
        # Generate a grid packing that is guaranteed valid
        # 26 circles. 5x5 grid has 25. 
        # Let's just place them in a 5x5 grid with radius 0.09 (safe)
        # And place the 26th somewhere?
        # Actually, let's just return the result and hope the optimizer worked.
        # If it failed, the validation function printed errors, but we must return something.
        # Let's try a safe grid.
        pass

    sum_radii = np.sum(radii_sol)
    
    # If invalid, fallback to a safe solution
    if not is_valid:
        # Generate safe 5x5 grid + 1
        # 5x5 grid radius 0.09 fits easily.
        # 26th circle?
        # Let's just do 26 circles of radius 0.09 in a grid-like pattern?
        # Or just return the calculated one, assuming errors are negligible.
        # But strict validation requires no overlap.
        # Let's try to fix overlaps by shrinking radii.
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_sol[i, 0] - centers_sol[j, 0]
                dy = centers_sol[i, 1] - centers_sol[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii_sol[i] + radii_sol[j]:
                    # Shrink larger one or both
                    excess = (radii_sol[i] + radii_sol[j]) - dist
                    radii_sol[i] -= excess/2
                    radii_sol[j] -= excess/2
        
        # Fix boundaries
        for i in range(n):
            x, y = centers_sol[i]
            radii_sol[i] = min(radii_sol[i], x, 1-x, y, 1-y)
            if radii_sol[i] < 0: radii_sol[i] = 0

        is_valid = validate_packing(centers_sol, radii_sol)
        sum_radii = np.sum(radii_sol)
        if not is_valid:
             # Last resort: very small circles
             radii_sol = np.ones(n) * 0.05
             # Place in 5x5 grid + 1
             # Just random positions?
             # Let's just return the current best effort.
             pass

    return centers_sol, radii_sol, sum_radii

# Execute and print result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")
