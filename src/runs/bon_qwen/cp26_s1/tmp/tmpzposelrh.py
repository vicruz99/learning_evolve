import numpy as np
import math
from scipy.optimize import minimize

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

def get_hex_grid_initialization(n_circles):
    """
    Generates a hexagonal grid of points for initialization.
    """
    # Estimate a starting radius. 
    # For N=26, area approx 1. If density ~0.9, 26 * pi * r^2 ~ 0.9 => r ~ 0.105
    # But boundary effects reduce this. Let's start with 0.09 to be safe and fit easily.
    r_start = 0.085 
    
    points = []
    row = 0
    while len(points) < n_circles:
        y = r_start + row * math.sqrt(3) * r_start
        if y + r_start > 1.0:
            # If we can't fit more rows with this radius, reduce radius slightly and retry?
            # Or just stop. But we need n_circles.
            # Let's adjust r_start dynamically if needed, but for init, fixed is easier.
            # If we run out of space, we just won't fill, but we need n_circles.
            # Let's just proceed and fill as much as possible, then optimizer will adjust.
            # Actually, better to scale down r_start if we hit boundary early.
            pass
        
        # X positions for this row
        if row % 2 == 0:
            start_x = r_start
        else:
            start_x = 2 * r_start # Shifted by one radius unit (half period 2r)
            
        col = 0
        while True:
            x = start_x + col * (2 * r_start)
            if x + r_start > 1.0:
                break
            
            points.append([x, y])
            col += 1
            if len(points) >= n_circles:
                break
        row += 1
        if row > 20: # Safety break
            break
            
    # If we didn't get enough points (unlikely with r=0.085), pad or scale
    # With r=0.085, width step 0.17. 1/0.17 ~ 5.8 circles per row. 
    # Height step ~ 0.147. 1/0.147 ~ 6.8 rows.
    # Total capacity > 26.
    
    return points[:n_circles], r_start

def run_packing():
    n = 26
    centers_init, r_init = get_hex_grid_initialization(n)
    
    # Initial vector: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    x0 = []
    for i in range(n):
        x0.extend([centers_init[i][0], centers_init[i][1], r_init])
    
    # Bounds for variables
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Constraints
    constraints = []
    
    # 1. Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        idx = 3 * i
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx] - v[idx+2]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx] - v[idx+2]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx+1] - v[idx+2]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx+1] - v[idx+2]
        })
        
    # 2. Non-overlap constraints: dist(c_i, c_j) >= r_i + r_j
    # dist - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = 3 * i
            idx_j = 3 * j
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i_idx=idx_i, j_idx=idx_j: \
                    np.sqrt((v[i_idx] - v[j_idx])**2 + (v[i_idx+1] - v[j_idx+1])**2) - v[i_idx+2] - v[j_idx+2]
            })
            
    # Objective: Maximize sum of radii => Minimize negative sum of radii
    def objective(v):
        radii_sum = 0
        for i in range(n):
            radii_sum += v[3*i + 2]
        return -radii_sum

    # Run optimization
    # Using SLSQP which handles constraints well
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        
        if result.success or result.fun > -2.0: # Heuristic check
            final_centers = np.zeros((n, 2))
            final_radii = np.zeros(n)
            for i in range(n):
                final_centers[i] = [result.x[3*i], result.x[3*i+1]]
                final_radii[i] = result.x[3*i+2]
            
            # Validate and clean up tiny negative radii if any (shouldn't happen due to bounds)
            final_radii = np.maximum(final_radii, 0)
            
            # Final check
            if validate_packing(final_centers, final_radii):
                total_sum = np.sum(final_radii)
                return final_centers, final_radii, total_sum
            else:
                # Fallback to initial if optimization failed validation
                # This shouldn't happen often with good start
                return np.array(centers_init), np.full(n, r_init), n * r_init
        else:
            # If optimization fails, return initial
            return np.array(centers_init), np.full(n, r_init), n * r_init
            
    except Exception:
        return np.array(centers_init), np.full(n, r_init), n * r_init