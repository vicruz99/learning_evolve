# sol_000217 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5b6844e7) state=08c1b501 sum of radii=2.513741 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def calculate_objective(x, n):
    """Calculate negative sum of radii."""
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    return -np.sum(radii)

def calculate_constraints(x, n):
    """Calculate constraints for optimization."""
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    constraints = []

    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        constraints.append(cx - r)
        constraints.append(1 - cx - r)
        constraints.append(cy - r)
        constraints.append(1 - cy - r)

    # Overlap constraints: dist(c1, c2) >= r1 + r2
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            constraints.append(dist - (radii[i] + radii[j]))

    return np.array(constraints)

def validate_packing(centers, radii):
    """Validate that circles don't overlap and are inside the unit square."""
    import math
    n = centers.shape[0]
    
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < -1e-12:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)

    # 1. Deterministic Hexagonal Initialization
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # Counts for 5 rows: 6, 5, 6, 5, 4
    row_counts = [6, 5, 6, 5, 4]
    current_idx = 0
    
    # Vertical spacing for hex packing is sqrt(3)/2 * diameter = sqrt(3) * r
    # We use a loose initial radius to avoid immediate constraint violations
    initial_r = 0.05
    dy = np.sqrt(3) * initial_r
    dx = 2 * initial_r
    
    current_y = initial_r
    for r_idx, count in enumerate(row_counts):
        # Stagger rows
        start_x = initial_r if r_idx % 2 == 0 else 2 * initial_r
        
        for c_idx in range(count):
            centers[current_idx, 0] = start_x + c_idx * dx
            centers[current_idx, 1] = current_y
            current_idx += 1
        current_y += dy

    x0 = np.hstack((centers.flatten(), radii))
    
    # 2. Numerical Optimization
    bounds = []
    for i in range(2 * n):
        bounds.append((0, 1)) # Centers in [0, 1]
    for i in range(n):
        bounds.append((0, 0.5)) # Radii > 0

    constraint_dict = {
        'type': 'ineq',
        'fun': lambda x: calculate_constraints(x, n)
    }

    res = minimize(
        calculate_objective,
        x0,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint_dict,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    # 3. Post-processing and Validation
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # Numerical cleanup for safety
    final_radii = np.maximum(final_radii, 0)
    
    if validate_packing(final_centers, final_radii):
        total_radius = np.sum(final_radii)
        return final_centers, final_radii, total_radius
    else:
        # Fallback to a safe grid if optimization fails validation
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.04
        idx = 0
        for y in range(6):
            for x in range(5):
                if idx < n:
                    centers[idx] = [0.1 + x * 0.18, 0.1 + y * 0.16]
                    idx += 1
        return centers, radii, np.sum(radii)
