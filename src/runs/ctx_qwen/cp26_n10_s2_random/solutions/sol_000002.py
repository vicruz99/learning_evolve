# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2a4ed9f3) state=2c120403 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

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

N = 26

def compute_overlap(p):
    """
    Computes overlap constraints for all pairs of circles.
    Constraint: dist^2 >= (r1 + r2)^2  =>  dist^2 - (r1 + r2)^2 >= 0
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Vectorized computation of pairwise squared distances and sum of radii
    # x, y, r are arrays of size N
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    sum_r_sq = dr**2
    
    # We only need constraints for i < j (upper triangle excluding diagonal)
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)
    return (dist_sq - sum_r_sq)[mask]

def compute_boundary(p):
    """
    Computes boundary constraints.
    Constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0, r >= 0
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r, r])

def objective(p):
    """
    Objective function: Minimize -sum(radii) to maximize sum of radii.
    """
    return -np.sum(p[2::3])

def get_initial_params():
    """
    Generates initial parameters for optimization using a hexagonal packing layout.
    """
    params = np.zeros(N * 3)
    idx = 0
    r = 0.09  # Initial radius guess
    y = r
    row = 0
    while idx < N:
        # Hexagonal packing: alternate rows are shifted
        start_x = r if row % 2 == 0 else 2 * r
        
        x = start_x
        while x + r <= 1.0 + 1e-9 and idx < N:
            params[3*idx] = x
            params[3*idx+1] = y
            params[3*idx+2] = r
            idx += 1
            x += 2 * r
        y += np.sqrt(3) * r
        row += 1
    return params

def run_packing() -> tuple:
    # Get initial guess
    p0 = get_initial_params()
    
    # Define bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    
    # Define constraints
    cons_overlap = opt.NonlinearConstraint(compute_overlap, 0, np.inf)
    cons_boundary = opt.NonlinearConstraint(compute_boundary, 0, np.inf)
    constraints = [cons_overlap, cons_boundary]
    
    # Run optimization
    try:
        result = opt.minimize(
            objective, 
            p0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
        )
        p_opt = result.x
    except Exception:
        p_opt = p0
    
    # Extract results
    centers = np.array([[p_opt[3*i], p_opt[3*i+1]] for i in range(N)])
    radii = p_opt[2::3].copy()
    
    # Repair function to ensure strict validity (handling numerical precision issues)
    for _ in range(50):
        changed = False
        # Check overlaps and reduce radii if needed
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    overlap = sum_r - dist
                    radii[i] -= overlap / 2
                    radii[j] -= overlap / 2
                    changed = True
        
        # Check boundaries and clamp radii
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if r < 0: 
                radii[i] = 0
                changed = True
            else:
                max_r = min(x, 1 - x, y, 1 - y)
                if r > max_r + 1e-12:
                    radii[i] = max_r
                    changed = True
        if not changed:
            break
            
    # Final validation
    if validate_packing(centers, radii):
        return centers, radii, np.sum(radii)
    else:
        # Fallback: 26 circles of radius 0.075 in a grid (safe valid packing)
        centers_fb = np.zeros((N, 2))
        radii_fb = np.full(N, 0.075)
        idx = 0
        r = 0.075
        # Grid layout: 6 rows, 5 cols
        for row in range(6):
            for col in range(5):
                if idx < N:
                    centers_fb[idx, 0] = r + col * 2 * r
                    centers_fb[idx, 1] = r + row * 2 * r
                    idx += 1
        return centers_fb, radii_fb, np.sum(radii_fb)
