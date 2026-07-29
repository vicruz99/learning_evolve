# sol_000361 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b037cf31) state=83dba712 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

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

def hex_grid_centers(n):
    """Generate centers based on a hexagonal grid pattern."""
    centers = []
    y = 0.1
    row = 0
    while len(centers) < n:
        if row % 2 == 0:
            x_start = 0.1
        else:
            x_start = 0.15 # Offset for hex pattern
        x = x_start
        while len(centers) < n and x < 0.95:
            centers.append([x, y])
            x += 0.2
        y += 0.1732 # sqrt(3)/2 * 0.2 approx
        row += 1
    return np.array(centers[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum_radii = -1.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    # Define constraints for SLSQP
    def constraints(n, x_vars):
        # x_vars layout: x1, y1, ..., xn, yn, r1, ..., rn
        centers = np.zeros((n, 2))
        centers[:, 0] = x_vars[:n]
        centers[:, 1] = x_vars[n:2*n]
        radii = x_vars[2*n:]
        
        c_list = []
        # Boundary constraints
        for i in range(n):
            # x >= r
            c_list.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
            # 1 - x >= r => x + r <= 1
            c_list.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*n+i]})
            # y >= r
            c_list.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
            # 1 - y >= r => y + r <= 1
            c_list.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[n+i] - x[2*n+i]})
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                def dist_con(x, i=i, j=j):
                    c1 = np.array([x[i], x[n+i]])
                    c2 = np.array([x[j], x[n+j]])
                    dist = np.linalg.norm(c1 - c2)
                    return dist - (x[2*n+i] + x[2*n+j])
                c_list.append({'type': 'ineq', 'fun': dist_con})
        return c_list

    def objective(x_vars):
        radii = x_vars[2*n:]
        return -np.sum(radii) # Minimize negative sum

    # Generate starting configurations
    starts = []
    
    # 1. Hexagonal Grid
    starts.append(hex_grid_centers(n))
    
    # 2. Perturbed Hex
    starts.append(hex_grid_centers(n) + np.random.uniform(-0.02, 0.02, (n, 2)))
    
    # 3. Random
    starts.append(np.random.uniform(0.05, 0.95, (n, 2)))
    
    # 4. Structured Grid
    cols = int(np.sqrt(n)) + 1
    grid = np.array([[0.1 + 0.8*(j/cols), 0.1 + 0.8*(i/(cols-1))] for i in range(cols) for j in range(cols)])[:n]
    starts.append(grid)

    # Optimize from each start
    for centers in starts:
        # Perturb slightly to ensure validity if needed
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial radii estimate
        radii_init = 0.02 * np.ones(n)
        
        # Build initial vector
        x0 = np.concatenate([centers[:, 0], centers[:, 1], radii_init])
        
        cons = constraints(n, x0)
        
        # Bounds for variables
        # x, y in [0, 1]
        # r in [0, 0.5]
        bnds = [(0, 1)] * (2*n) + [(0, 0.5)] * n

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, 
                           options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or res.fun < -best_sum_radii:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = np.column_stack([res.x[:n], res.x[n:2*n]])
                    best_radii = res.x[2*n:]
        except Exception:
            continue

    # Validate best found
    if validate_packing(best_centers, best_radii):
        return best_centers, best_radii, float(best_sum_radii)
    else:
        # Fallback to a simple valid packing if optimizer fails validation
        # 5x5 grid + 1 small circle in center
        centers_fallback = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for j in range(5) for i in range(5)] + [[0.5, 0.5]])
        radii_fallback = [0.09]*25 + [0.02]
        return centers_fallback[:n], np.array(radii_fallback[:n]), float(sum(radii_fallback[:n]))
