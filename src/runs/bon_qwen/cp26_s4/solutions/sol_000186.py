# sol_000186 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 083f9270) state=a1e05535 sum of radii=1.734423 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def get_max_radii_sum(centers):
    """
    Solves the Linear Programming problem to find the optimal radii 
    for a fixed set of centers to maximize the sum of radii.
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c = -np.ones(n)

    # Inequality constraints: A_ub @ r <= b_ub
    A_ub = []
    b_ub = []

    # 1. Boundary constraints: r_i <= dist to boundary
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        # r_i <= max_r
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(max_r)

    # 2. Pairwise constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds for radii (non-negative)
    bounds = [(0, None) for _ in range(n)]

    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def objective_neg_sum_radii(centers_flat):
    """
    Objective function for the center optimizer.
    """
    centers = centers_flat.reshape(-1, 2)
    total_sum, radii = get_max_radii_sum(centers)
    return -total_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers with a hexagonal grid
    # Spacing chosen to fit ~26 circles
    s = 0.22 
    centers_list = []
    rows = 6
    cols = 5
    
    for r in range(rows):
        for c in range(cols):
            x = c * s + (r % 2) * (s / 2)
            y = r * s * np.sqrt(3) / 2
            if len(centers_list) < n:
                centers_list.append([x, y])

    # Trim to exactly 26 and center/scale to unit square
    initial_centers = np.array(centers_list[:n])
    
    # Normalize to [0, 1]
    min_c = initial_centers.min(axis=0)
    max_c = initial_centers.max(axis=0)
    span = max_c - min_c
    # Add small padding to avoid exact boundary issues
    initial_centers = (initial_centers - min_c) / span
    initial_centers = initial_centers * 0.95 + 0.025

    # 2. Optimize centers
    # Using Powell method which is robust for non-smooth functions
    bounds_centers = [(0, 1)] * (n * 2)
    result = minimize(objective_neg_sum_radii, initial_centers.flatten(), 
                      method='Powell', bounds=bounds_centers)

    # 3. Final validation and radii calculation
    final_centers = result.x.reshape(-1, 2)
    final_sum, final_radii = get_max_radii_sum(final_centers)

    return final_centers, final_radii, final_sum
