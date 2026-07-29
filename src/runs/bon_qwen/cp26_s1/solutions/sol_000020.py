# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f223c9a2) state=230422e9 sum of radii=1.300000 correctness=1.0
# stdout(first 200): Circles 0 and 1 overlap: dist=4.055734516585436e-15, r1+r2=1.0 Circles 0 and 1 overlap: dist=1.8452761694870047e-15, r1+r2=0.9999999999999705 Circles 0 and 1 overlap: dist=2.0014830212433605e-15, r1+r
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def hexagonal_grid_init(n_circles, rows, radius):
    """Generates a hexagonal grid of circle centers."""
    centers = []
    count = 0
    for row in range(rows):
        # Stagger x-coordinate for odd rows
        x_start = radius if row % 2 == 0 else 2 * radius
        y_coord = radius + row * radius * np.sqrt(3)
        
        # Determine number of circles in this row to fit width
        # Width available is 1. Circle width is 2r.
        # x + 2r <= 1.
        while True:
            x_coord = x_start + (len([c for c in centers if c[1] == y_coord])) * 2 * radius
            if x_coord + radius <= 1.0 + 1e-9:
                centers.append((x_coord, y_coord))
                count += 1
                if count >= n_circles:
                    break
            else:
                break
        if count >= n_circles:
            break
    return np.array(centers[:n_circles])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_result = None
    best_sum = -1.0

    # Generate multiple initial guesses to avoid local minima
    # Heuristic: 26 circles, roughly 6 rows of hexagonal packing
    configs_to_try = []
    
    # Configuration 1: Hexagonal with estimated radius
    # 10r approx 1 -> r=0.1 is tight for 5 cols. 
    # Let's try to fit them with a slightly smaller radius first.
    r_init = 0.095 
    try:
        centers_init1 = hexagonal_grid_init(n, 8, r_init)
        configs_to_try.append(centers_init1)
    except:
        pass

    # Configuration 2: Uniform Grid (Square packing)
    # 6x5 grid = 30 cells. Pick 26.
    # Spacing 1/6 approx 0.166. r approx 0.08.
    x_grid = np.linspace(0.09, 0.91, 6)
    y_grid = np.linspace(0.1, 0.9, 5) # 5 rows
    grid_points = []
    for y in y_grid:
        for x in x_grid:
            grid_points.append([x, y])
    grid_points = np.array(grid_points[:n])
    configs_to_try.append(grid_points)
    
    # Configuration 3: Randomized hexagonal (perturbed)
    try:
        centers_init3 = hexagonal_grid_init(n, 7, 0.09)
        centers_init3 += np.random.normal(0, 0.01, centers_init3.shape)
        configs_to_try.append(centers_init3)
    except:
        pass

    for centers_guess in configs_to_try:
        r_guess = np.full(n, 0.01) # Start small to ensure feasibility
        
        # Initial variable vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = []
        for i in range(n):
            x0.extend([centers_guess[i, 0], centers_guess[i, 1], r_guess[i]])
        x0 = np.array(x0)

        # Bounds: x, y in [0, 1], r >= 0 (and effectively <= 0.5)
        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

        # Constraints
        cons = []
        
        # Boundary containment constraints: r <= x <= 1-r  => x - r >= 0, x + r <= 1
        for i in range(n):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # 1 - x - r >= 0  => x + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
                idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
                
                def constraint_overlap(v, i=i, j=j):
                    xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                    xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    rad_sum_sq = (ri + rj)**2
                    return dist_sq - rad_sum_sq
                
                cons.append({'type': 'ineq', 'fun': constraint_overlap})

        # Objective: Maximize sum of radii => Minimize -sum(r)
        def objective(v):
            radii = v[2::3]
            return -np.sum(radii)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'ftol': 1e-12, 'maxiter': 2000, 'disp': False})
            
            if res.success:
                centers_opt = np.zeros((n, 2))
                radii_opt = np.zeros(n)
                for i in range(n):
                    centers_opt[i, 0] = res.x[3*i]
                    centers_opt[i, 1] = res.x[3*i+1]
                    radii_opt[i] = res.x[3*i+2]
                
                # Check validity manually to be safe
                if validate_packing(centers_opt, radii_opt):
                    current_sum = np.sum(radii_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = (centers_opt, radii_opt)
        except Exception as e:
            continue

    if best_result is None:
        # Fallback to a simple valid packing if optimization fails
        # Place circles in a grid
        centers_fb = np.zeros((n, 2))
        r_fb = 0.05
        # Just fill grid
        idx = 0
        for r_idx in range(6):
            for c_idx in range(5):
                if idx < n:
                    centers_fb[idx] = [0.1 + c_idx * 0.2, 0.1 + r_idx * 0.15]
                    idx += 1
        radii_fb = np.full(n, r_fb)
        return centers_fb, radii_fb, np.sum(radii_fb)

    return best_result[0], best_result[1], best_sum

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
