# sol_000261 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3433cac4) state=bdffb7ca sum of radii=2.501000 correctness=1.0
# stdout(first 200): Circles 1 and 2 overlap: dist=0.19234836826519353, r1+r2=0.19234836826629892 Circles 1 and 2 overlap: dist=0.19234836826519353, r1+r2=0.19234836826629892 Circles 1 and 2 overlap: dist=0.19234836826519
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

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

def hexagonal_grid(n, seed=0):
    np.random.seed(seed)
    centers = []
    if n <= 0:
        return np.array(centers)
    
    # Heuristic to place points in a hexagonal pattern
    # We try to fit as many as possible in a square unit cell
    # Adjusting row spacing and horizontal shift for hexagonal packing
    
    # Try to determine number of rows and cols
    # For hex packing, row spacing is sqrt(3)/2 * 2r approx 1.732 r
    # We don't know r, but we can scale later. Let's assume r=0.1 for layout
    r_layout = 0.1
    dy = math.sqrt(3) * r_layout
    dx = 2 * r_layout
    
    y = r_layout
    row_idx = 0
    
    while len(centers) < n:
        # Offset every other row
        x_start = r_layout + (row_idx % 2) * dx / 2
        x = x_start
        while x <= 1 - r_layout and len(centers) < n:
            centers.append([x, y])
            x += dx
        y += dy
        row_idx += 1
        
        # Safety break if we are going out of bounds significantly
        if y > 1 + r_layout * 2: 
            break
            
    centers = np.array(centers)
    
    # If we didn't reach n, pad with random
    if len(centers) < n:
        diff = n - len(centers)
        random_adds = np.random.rand(diff, 2)
        centers = np.vstack([centers, random_adds])
        
    return centers[:n]

def objective(vars, n):
    # vars = [x1, y1, ..., xn, yn, r]
    # We want to maximize r, so we minimize -r
    return -vars[-1]

def constraints_builder(vars, n):
    constraints = []
    centers = vars[:2*n].reshape(n, 2)
    r = vars[-1]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    # x - r >= 0
    # 1 - x - r >= 0
    for i in range(n):
        x, y = centers[i]
        # x - r >= 0
        constraints.append(x - r)
        # 1 - x - r >= 0
        constraints.append(1 - x - r)
        # y - r >= 0
        constraints.append(y - r)
        # 1 - y - r >= 0
        constraints.append(1 - y - r)
        
    # Distance constraints: dist_ij >= 2r
    # dist^2 - 4r^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            c1 = centers[i]
            c2 = centers[j]
            dist_sq = np.sum((c1 - c2)**2)
            constraints.append(dist_sq - 4*r**2)
            
    return np.array(constraints)

def run_packing() -> tuple:
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    seeds = [0, 1, 2, 3, 4, 5, 10, 20, 100]
    
    for seed in seeds:
        # Initial guess
        centers_init = hexagonal_grid(n, seed)
        # Scale to fit in unit square roughly
        centers_init = centers_init - centers_init.min(axis=0)
        centers_init = centers_init / (centers_init.max(axis=0) - centers_init.min(axis=0) + 1e-8)
        centers_init = centers_init * 0.8 + 0.1 # Add padding
        
        r_init = 0.05
        x0 = np.concatenate([centers_init.flatten(), [r_init]])
        
        # Constraints for SLSQP
        # fun(x) >= 0
        cons = {'type': 'ineq', 'fun': lambda v: constraints_builder(v, n)}
        
        # Bounds for centers [0, 1] and r [0, 1]
        bounds = [(0, 1)] * (2 * n) + [(0, 1)]
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -0.09: # If radius > 0.09
                r_opt = res.x[-1]
                c_opt = res.x[:-1].reshape(n, 2)
                
                # Validate and adjust if needed
                if validate_packing(c_opt, np.full(n, r_opt)):
                    current_sum = np.sum(np.full(n, r_opt))
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = c_opt
                        best_radii = np.full(n, r_opt)
        except Exception as e:
            continue

    # Fallback if optimization fails or finds poor solution
    if best_sum < 0.01:
        # Provide a simple valid packing (e.g., 5x5 grid for 25, 1 small)
        best_centers = np.array([]).reshape(0, 2)
        # 5x5 grid
        coords = np.linspace(0.1, 0.9, 5)
        cx, cy = np.meshgrid(coords, coords)
        grid = np.column_stack([cx.ravel(), cy.ravel()])
        best_centers = grid
        
        # Add 26th circle at a safe spot with small radius
        # Center of square might be taken, try (0.1, 0.1) is taken.
        # Try (0.5, 0.5) is taken in 5x5? 
        # 5x5 grid points: 0.1, 0.3, 0.5, 0.7, 0.9. 
        # (0.5, 0.5) is a center.
        # Let's just put the 26th circle at (0.01, 0.01) with r=0.01
        best_centers = np.vstack([best_centers, [0.01, 0.01]])
        best_radii = np.array([0.1]*25 + [0.001])
        best_sum = np.sum(best_radii)
        # Re-validate
        if not validate_packing(best_centers, best_radii):
             # Try to shrink slightly
             best_radii[:] *= 0.99
             best_sum = np.sum(best_radii)

    # Final validation
    if validate_packing(best_centers, best_radii):
        return best_centers, best_radii, float(np.sum(best_radii))
    else:
        # Emergency shrink
        factor = 0.95
        while not validate_packing(best_centers, best_radii * factor):
            factor *= 0.95
        best_radii *= factor
        return best_centers, best_radii, float(np.sum(best_radii))
