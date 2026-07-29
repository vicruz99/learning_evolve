# sol_000219 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 76d635d8) state=58dca5f1 sum of radii=2.614732 correctness=1.0
# stdout(first 200): Circle 0 at (0.07359368201676589, 0.07359368201391484) with radius 0.07359368201570131 is outside the unit square
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_hexagonal_points(n, seed=42):
    """
    Generate n points in a hexagonal lattice pattern, 
    scaled and perturbed to fit in [0,1]x[0,1].
    """
    np.random.seed(seed)
    
    # Estimate grid size
    cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    rows = int(np.ceil(n / cols))
    
    points = []
    # Generate points in a hexagonal grid
    for r in range(rows):
        for c in range(cols):
            if len(points) >= n:
                break
            # Hexagonal offset
            x = c * np.sqrt(3)
            y = r * 1.5 + (c % 2) * 0.75
            points.append([x, y])
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    
    # Normalize to fit in unit square
    if points.size > 0:
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        # Avoid division by zero
        range_x = max_x - min_x if max_x > min_x else 1
        range_y = max_y - min_y if max_y > min_y else 1
        
        # Scale to 0.8 size to leave margin
        scale = 0.8 / max(range_x, range_y)
        points[:, 0] = (points[:, 0] - min_x) * scale + 0.1
        points[:, 1] = (points[:, 1] - min_y) * scale + 0.1
        
        # Add small random perturbation
        points += np.random.uniform(-0.02, 0.02, points.shape)
        points = np.clip(points, 0.05, 0.95)
        
    return points

def objective(vars_flat, n):
    """Objective function: minimize negative sum of radii"""
    r = vars_flat[2::3]
    return -np.sum(r)

def boundary_constraints(vars_flat, n):
    """Constraints for circles staying inside the unit square"""
    constraints = []
    for i in range(n):
        x, y, r = vars_flat[3*i], vars_flat[3*i+1], vars_flat[3*i+2]
        # x - r >= 0
        constraints.append(x - r)
        # 1 - x - r >= 0
        constraints.append(1 - x - r)
        # y - r >= 0
        constraints.append(y - r)
        # 1 - y - r >= 0
        constraints.append(1 - y - r)
    return np.array(constraints)

def overlap_constraints(vars_flat, n):
    """Constraints for non-overlapping circles"""
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = vars_flat[3*i], vars_flat[3*i+1]
            xj, yj = vars_flat[3*j], vars_flat[3*j+1]
            ri, rj = vars_flat[3*i+2], vars_flat[3*j+2]
            
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r = ri + rj
            
            # dist >= sum_r  =>  dist^2 >= sum_r^2
            # dist_sq - sum_r^2 >= 0
            constraints.append(dist_sq - sum_r**2)
    return np.array(constraints)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Try multiple seeds to find a better local optimum
    seeds = [10, 20, 30, 40, 50]
    
    for seed in seeds:
        centers_init = get_hexagonal_points(n, seed=seed)
        radii_init = np.full(n, 0.05) # Small initial radius
        
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        # Define constraints for SLSQP
        cons = []
        
        # Boundary constraints (inequality >= 0)
        cons.append({'type': 'ineq', 'fun': lambda x: boundary_constraints(x, n)})
        
        # Overlap constraints (inequality >= 0)
        cons.append({'type': 'ineq', 'fun': lambda x: overlap_constraints(x, n)})
        
        # Bounds for variables
        # x, y in [0, 1], r >= 0
        bounds = []
        for _ in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 1)) # r (radius cannot exceed 0.5 in unit square)
            
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
            
            if res.success or (res.fun > best_sum):
                # Validate manually to be safe before updating best
                current_centers = res.x.reshape(-1, 3)[:, :2]
                current_radii = res.x.reshape(-1, 3)[:, 2]
                
                # Quick check for obvious invalidity (NaNs, negatives)
                if not np.isnan(current_centers).any() and not np.isnan(current_radii).any():
                    if np.all(current_radii >= 0):
                        sum_r = np.sum(current_radii)
                        if sum_r > best_sum:
                            # Full validation
                            if validate_packing(current_centers, current_radii):
                                best_sum = sum_r
                                best_centers = current_centers
                                best_radii = current_radii
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback to a simple grid if optimization fails
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                    radii[idx] = 0.09
                    idx += 1
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    return best_centers, best_radii, best_sum

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
