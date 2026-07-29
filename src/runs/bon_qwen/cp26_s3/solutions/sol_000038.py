# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b444b7b1) state=55c376c1 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    import math
    n = centers.shape[0]

    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # 1. Initial placement based on strategy
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Large central circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.5

    # Four corner circles (tangent to center and walls)
    r_corner = (np.sqrt(2)/2 - 0.5) / (1 + np.sqrt(2))
    radii[1:5] = r_corner
    centers[1] = [r_corner, r_corner]
    centers[2] = [1-r_corner, r_corner]
    centers[3] = [1-r_corner, 1-r_corner]
    centers[4] = [r_corner, 1-r_corner]

    # Eight edge fillers (between corners)
    r_edge = 0.06
    radii[5:13] = r_edge
    centers[5] = [0.5, r_edge]
    centers[6] = [r_edge, 0.5]
    centers[7] = [0.5, 1-r_edge]
    centers[8] = [1-r_edge, 0.5]
    centers[9] = [0.5-r_edge, 0.5-r_edge]
    centers[10] = [0.5+r_edge, 0.5-r_edge]
    centers[11] = [0.5-r_edge, 0.5+r_edge]
    centers[12] = [0.5+r_edge, 0.5+r_edge]

    # Thirteen internal fillers
    r_int = 0.04
    radii[13:26] = r_int
    for i in range(13, 26):
        centers[i] = [0.2 + 0.3*(i-13), 0.2 + 0.3*(i-13) % 4]

    # 2. Optimization via penalty method
    # Convert to flat array for scipy
    params = np.concatenate([centers.flatten(), radii])

    def objective(x):
        return -np.sum(x[52:]) # Maximize sum of radii

    def constraint_func(x):
        c = x[:52].reshape(26, 2)
        r = x[52:]
        # Boundary constraints
        cons = []
        for i in range(n):
            cons.append(c[i, 0] - r[i])
            cons.append(1 - c[i, 0] - r[i])
            cons.append(c[i, 1] - r[i])
            cons.append(1 - c[i, 1] - r[i])
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                cons.append(dist - (r[i] + r[j]))
        return np.array(cons)

    constraints = {'type': 'ineq', 'fun': constraint_func}
    bounds = [(0, 1) if i < 52 else (0, 1) for i in range(78)]

    res = minimize(objective, params, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})

    centers_opt = res.x[:52].reshape(26, 2)
    radii_opt = res.x[52:]

    # 3. Safety shrink to satisfy strict tolerance
    max_r_shrink = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
            required = radii_opt[i] + radii_opt[j]
            if dist < required:
                overlap = (required - dist) / 2
                if overlap > max_r_shrink:
                    max_r_shrink = overlap
    
    # Also check boundaries
    for i in range(n):
        r = radii_opt[i]
        c = centers_opt[i]
        margin = min(c[0], 1-c[0], c[1], 1-c[1])
        if margin < r:
            max_r_shrink = max(max_r_shrink, r - margin)

    if max_r_shrink > 0:
        radii_opt *= (1 - 2 * max_r_shrink - 1e-12)
        if any(radii_opt < 0):
            radii_opt[radii_opt < 0] = 0

    return centers_opt, radii_opt, np.sum(radii_opt)
