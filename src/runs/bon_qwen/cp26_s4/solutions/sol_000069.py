# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 10bf7585) state=224770d5 sum of radii=2.486412 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (centers, radii, sum_radii) for 26 circles in a unit square.
    """
    n_circles = 26
    np.random.seed(42)

    # 1. Initialization: 5x5 grid + 1 center
    centers = np.zeros((n_circles, 2))
    grid_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    idx = 0
    for x in grid_points:
        for y in grid_points:
            centers[idx] = [x, y]
            idx += 1
    # Add 26th circle in the center (perturbed slightly)
    centers[idx] = [0.5, 0.5]
    
    # Initial radii estimate
    radii = np.full(n_circles, 0.1)

    # Objective coefficients for LP: minimize sum of (-r)
    c_obj = -np.ones(n_circles)
    
    # Bounds for LP variables (radii >= 0)
    bounds = [(0, None) for _ in range(n_circles)]

    best_sum_radii = 0.0
    best_centers = centers.copy()
    best_radii = radii.copy()

    # Precompute pair indices to avoid loops during optimization
    pair_indices = [(i, j) for i in range(n_circles) for j in range(i + 1, n_circles)]
    n_pairs = len(pair_indices)
    n_constraints = 4 * n_circles + n_pairs

    for step in range(2000):
        # Step 1: Solve LP for optimal radii given current centers
        A_ub = np.zeros((n_constraints, n_circles))
        b_ub = np.zeros(n_constraints)

        # Boundary constraints: r <= min(x, 1-x, y, 1-y)
        # 4 constraints per circle
        for i in range(n_circles):
            x, y = centers[i]
            # r_i <= x  =>  1*r_i <= x
            A_ub[4 * i, i] = 1.0
            b_ub[4 * i] = x
            # r_i <= 1 - x => 1*r_i <= 1 - x
            A_ub[4 * i + 1, i] = 1.0
            b_ub[4 * i + 1] = 1.0 - x
            # r_i <= y
            A_ub[4 * i + 2, i] = 1.0
            b_ub[4 * i + 2] = y
            # r_i <= 1 - y
            A_ub[4 * i + 3, i] = 1.0
            b_ub[4 * i + 3] = 1.0 - y

        # Pairwise constraints: r_i + r_j <= distance(c_i, c_j)
        row = 4 * n_circles
        for i, j in pair_indices:
            dist = np.linalg.norm(centers[i] - centers[j])
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = dist
            row += 1

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
                current_sum = -res.fun
                
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
        except Exception:
            continue

        # Step 2: Compute forces to move centers
        forces = np.zeros_like(centers)
        tol = 1e-4

        # Forces from walls
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            # Left wall
            if r > x - tol:
                forces[i, 0] += 1.0
            # Right wall
            if r > (1 - x) - tol:
                forces[i, 0] -= 1.0
            # Bottom wall
            if r > y - tol:
                forces[i, 1] += 1.0
            # Top wall
            if r > (1 - y) - tol:
                forces[i, 1] -= 1.0

        # Forces from neighbors (repulsion if touching)
        for i, j in pair_indices:
            r_sum = radii[i] + radii[j]
            dist = np.linalg.norm(centers[i] - centers[j])
            
            # If touching (within tolerance)
            if dist < r_sum + tol and dist > 1e-9:
                # Unit vector from j to i
                direction = (centers[i] - centers[j]) / dist
                forces[i] += direction
                forces[j] -= direction

        # Step 3: Update centers
        # Decay step size over time
        alpha = 0.05 / (1.0 + 0.005 * step)
        centers += alpha * forces

        # Clip centers to valid region to avoid singularity
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)

    return best_centers, best_radii, best_sum_radii
