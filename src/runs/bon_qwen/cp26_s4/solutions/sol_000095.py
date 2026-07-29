# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e1ebaf70) state=52c3535c sum of radii=0.260000 correctness=1.0
# stdout(first 200): Circle 0 at (0.5940894120008003, 0.8493018325258501) with radius 0.49999999999999967 is outside the unit square Validation failed. Attempting fallback to grid.
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal-like lattice (Rows of 6, 5, 6, 5, 4)
    # This creates a dense initial packing.
    r_init = 0.05
    centers_init = []
    radii_init = []
    
    # Row configurations: (number of circles, x-offset)
    # Offset 0: aligned with left edge, Offset 1: shifted by r_init
    row_config = [
        (6, 0), (5, 1), (6, 0), (5, 1), (4, 0)
    ]
    
    current_row_idx = 0
    for num_circles, offset_type in row_config:
        y = r_init + current_row_idx * np.sqrt(3) * r_init
        base_x = r_init + (offset_type * r_init)
        
        for k in range(num_circles):
            x = base_x + k * (2 * r_init)
            centers_init.append([x, y])
            radii_init.append(r_init)
        
        current_row_idx += 1
    
    centers_init = np.array(centers_init)
    radii_init = np.array(radii_init)
    
    # 2. Define optimization variables and objective
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.hstack([centers_init.flatten(), radii_init])
    
    def objective(vars):
        # Maximize sum of radii => Minimize negative sum
        r = vars[2::3]
        return -np.sum(r)
    
    # 3. Define constraints
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        idx_x = i * 3
        idx_y = i * 3 + 1
        idx_r = i * 3 + 2
        
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[idx_x] - v[idx_r]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[idx_y] - v[idx_r]
        })
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = i*3, i*3+1, i*3+2
            idx_xj, idx_yj, idx_rj = j*3, j*3+1, j*3+2
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    np.sqrt((v[i*3] - v[j*3])**2 + (v[i*3+1] - v[j*3+1])**2) - v[i*3+2] - v[j*3+2]
            })

    # Bounds: x, y in [0, 1], r >= 0
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5) for _ in range(n)]

    # 4. Run Optimization
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if res.success or res.fun < -2.5: # If we found a reasonably good solution
            final_centers = res.x[:2*n].reshape((n, 2))
            final_radii = res.x[2*n:]
        else:
            # Fallback to initial if optimization fails
            final_centers = centers_init
            final_radii = radii_init
            
    except Exception:
        final_centers = centers_init
        final_radii = radii_init

    # 5. Ensure non-negative radii (clip tiny negatives due to numerical error)
    final_radii = np.maximum(final_radii, 0.0)

    # 6. Validation check
    if not validate_packing(final_centers, final_radii):
        print("Validation failed. Attempting fallback to grid.")
        # Fallback: Simple grid
        step = 1.0 / 5.0
        fallback_centers = []
        for r in range(5):
            for c in range(5):
                fallback_centers.append([step/2 + c*step, step/2 + r*step])
        # Add 26th circle in center of first gap if possible
        fallback_centers.append([step/2 + step/2, step/2 + step/2]) # Center of first cell
        
        # Pad to 26
        while len(fallback_centers) < 26:
            fallback_centers.append([0.5, 0.5])
        fallback_centers = np.array(fallback_centers[:26])
        final_radii = np.ones(26) * 0.01
        final_centers = fallback_centers

    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii
