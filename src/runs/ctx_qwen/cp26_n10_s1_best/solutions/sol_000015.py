# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ad74c980) state=8ab04ca2 sum of radii=2.340149 correctness=1.0
# stdout(first 200): Circles 3 and 25 overlap: dist=0.16006584364166782, r1+r2=0.1600659245743289
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

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

def solve_lp(centers, n):
    """
    Solves the Linear Programming problem to maximize sum of radii for fixed centers.
    Maximize sum(r_i) subject to:
      r_i >= 0
      r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      r_i + r_j <= distance(c_i, c_j)
    """
    # Objective: minimize -sum(r_i) => c = [-1, -1, ..., -1]
    c = np.ones(n) * -1.0

    # Constraints matrix A_ub * r <= b_ub
    # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    # 2. Pairwise constraints: r_i + r_j <= dist_ij
    
    # We will build A_ub and b_ub lists
    A_ub = []
    b_ub = []

    for i in range(n):
        xi, yi = centers[i]
        
        # r_i <= xi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(xi)
        
        # r_i <= 1 - xi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - xi)
        
        # r_i <= yi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(yi)
        
        # r_i <= 1 - yi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - yi)

    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds for r_i: [0, 1] (1 is loose upper bound)
    bounds = [(0, 1) for _ in range(n)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.fun * -1, res.x # Return sum_radii, radii
        else:
            # Fallback to small radii if LP fails
            return 0.0, np.full(n, 1e-6)
    except Exception:
        return 0.0, np.full(n, 1e-6)

def get_sum_radii_for_centers(centers):
    """Wrapper to get just the sum of radii for a configuration"""
    n = centers.shape[0]
    s, _ = solve_lp(centers, n)
    return s

def optimize_single_center(centers, k, n):
    """
    Optimizes the position of the k-th circle center to maximize sum of radii.
    """
    other_centers = np.delete(centers, k, axis=0)
    
    def objective(coords):
        # coords is (x, y) for center k
        new_centers = np.vstack([
            centers[:k], 
            np.array(coords), 
            centers[k+1:]
        ])
        s, _ = solve_lp(new_centers, n)
        return -s # Minimize negative sum

    x0 = centers[k]
    
    # Use Nelder-Mead as it doesn't require gradients
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100})
    
    if res.success:
        return res.x
    else:
        return x0

def run_packing():
    n = 26
    # 1. Initialization: Hexagonal Grid
    # We want roughly 5x5 grid.
    # Let's try to place points in a hexagonal pattern.
    centers = []
    
    # Approximate spacing for 26 circles in unit square
    # Area per circle approx 1/26 ~ 0.038. Radius ~ 0.1. Diameter 0.2.
    # Hexagonal spacing s ~ diameter.
    s = 0.22 
    
    # Generate points
    row_idx = 0
    while len(centers) < n:
        col_idx = 0
        y = row_idx * s * np.sqrt(3)/2
        if y + s > 1.0: # Stop if y goes out
            break
            
        while len(centers) < n:
            x = col_idx * s
            if row_idx % 2 == 1:
                x += s / 2.0
            
            if x + s > 1.0:
                break
            
            # Check if point is within bounds [0,1]
            # We place center at (x+s/2, y+s/2) effectively to leave margin? 
            # No, let's just place at x, y and let optimization fix boundaries.
            # But better to keep inside.
            # Let's clamp or just add.
            # Actually, simple grid is fine, optimization will push them out.
            # Let's add a small margin to start.
            cx = x + 0.05
            cy = y + 0.05
            
            if cx <= 1.0 and cy <= 1.0:
                centers.append([cx, cy])
            
            col_idx += 1
        row_idx += 1
        
    centers = np.array(centers[:n])
    
    # Ensure we have exactly 26
    if len(centers) < n:
        # Fill remaining with random valid points if grid failed (unlikely)
        while len(centers) < n:
            centers = np.vstack([centers, np.random.rand(1, 2)])

    # 2. Coordinate Descent Optimization
    # Iterate a few times to refine positions
    num_iterations = 3
    
    for iteration in range(num_iterations):
        for k in range(n):
            # Optimize center k
            best_pos = optimize_single_center(centers, k, n)
            centers[k] = best_pos
            
    # 3. Final LP to get radii
    total_sum, radii = solve_lp(centers, n)
    
    # 4. Validation
    if validate_packing(centers, radii):
        return centers, radii, total_sum
    else:
        # Fallback: shrink radii slightly to satisfy constraints strictly if validation fails
        # This handles potential numerical edge cases
        for i in range(n):
            radii[i] *= 0.99
        total_sum = np.sum(radii)
        # Re-validate
        if not validate_packing(centers, radii):
             # If still fails, force valid state by clipping
             # This is a safety net
             for i in range(n):
                 x, y = centers[i]
                 max_r = min(x, 1-x, y, 1-y)
                 radii[i] = min(radii[i], max_r * 0.9)
             for i in range(n):
                 for j in range(i+1, n):
                     dist = np.linalg.norm(centers[i] - centers[j])
                     if radii[i] + radii[j] > dist:
                         scale = dist / (radii[i] + radii[j])
                         radii[i] *= scale * 0.99
                         radii[j] *= scale * 0.99
             total_sum = np.sum(radii)
             
        return centers, radii, total_sum

if __name__ == "__main__":
    centers, radii, total_sum = run_packing()
    print(f"Sum of radii: {total_sum}")
    print(f"Centers shape: {centers.shape}")
    print(f"Radii shape: {radii.shape}")
