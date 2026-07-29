# sol_000337 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 29661f66) state=07fd7487 sum of radii=0.448276 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import scipy.sparse as sp

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions of 26 circles and then determines the optimal radii
    using Linear Programming to maximize the sum of radii.
    """
    n = 26
    rng = np.random.default_rng(42)

    # 1. Initialize centers in a hexagonal grid pattern
    # We fit 26 points by selecting a subset of a slightly denser grid
    points = []
    row_spacing = 0.12
    col_spacing = row_spacing * np.sqrt(3)
    
    # Generate enough points and pick the first 26 that fit well
    y_val = row_spacing
    while len(points) < n:
        x_val = col_spacing / 2 if int(len(points) / (n // 4 + 1)) % 2 == 1 else 0
        # Generate a row
        row_points = []
        x_curr = x_val
        while x_curr < 1.0:
            row_points.append([x_curr, y_val])
            x_curr += col_spacing
        points.extend(row_points)
        y_val += row_spacing
    
    # Normalize or shift if needed to fit tightly, then trim to n
    centers_init = np.array(points[:n])
    
    # Center and scale to fit nicely in [0, 1]
    # Shift to center of square and scale to be slightly loose initially
    centers_init = centers_init - centers_init.mean(axis=0) + 0.5
    centers_init = centers_init * 0.9 + 0.05 * (1.0 - 0.9) # slight offset

    # 2. Define the objective function for center optimization
    def evaluate_centers(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        
        # Calculate pairwise distances
        # D_ij = distance between center i and center j
        # We only need upper triangular part for constraints
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            diff = centers[i, np.newaxis, :] - centers[np.newaxis, i, :]
            dist_matrix[i, i:] = np.sqrt(np.sum(diff**2, axis=1))

        # Setup LP: Maximize sum(r)
        # Subject to:
        # 1. r_i >= 0 (handled by bounds)
        # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # 3. r_i + r_j <= dist_matrix[i, j] for all i < j

        c_obj = -np.ones(n) # We minimize negative sum

        # Inequality constraints A_ub * r <= b_ub
        constraints_rows = []
        constraints_data = []
        constraints_b = []

        # Wall constraints
        for i in range(n):
            x, y = centers[i]
            # r_i <= x
            constraints_rows.extend([i])
            constraints_data.extend([1.0])
            constraints_b.append(x)
            
            # r_i <= 1 - x
            constraints_rows.extend([i])
            constraints_data.extend([1.0])
            constraints_b.append(1.0 - x)
            
            # r_i <= y
            constraints_rows.extend([i])
            constraints_data.extend([1.0])
            constraints_b.append(y)
            
            # r_i <= 1 - y
            constraints_rows.extend([i])
            constraints_data.extend([1.0])
            constraints_b.append(1.0 - y)

        # Pairwise constraints: r_i + r_j <= dist
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                constraints_rows.extend([i, j])
                constraints_data.extend([1.0, 1.0])
                constraints_b.append(d)

        # Construct sparse matrix
        rows = np.array(constraints_rows)
        cols = np.array(constraints_rows) # columns correspond to variables r_i
        A_ub = sp.csr_matrix((constraints_data, (rows, cols)), shape=(len(constraints_b), n))
        b_ub = np.array(constraints_b)

        # Solve LP
        try:
            res = scipy.optimize.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, 
                                        bounds=[(0, None)] * n, method='highs')
            if res.success:
                return -res.fun # Return sum of radii
            else:
                return 0.0
        except Exception:
            return 0.0

    # 3. Optimize centers using Nelder-Mead
    x0 = centers_init.flatten()
    bounds = [(0.0, 1.0) for _ in range(n * 2)]
    
    # Using Nelder-Mead as it handles non-convex landscapes reasonably well
    # We maximize, so we minimize negative
    res_opt = scipy.optimize.minimize(
        lambda x: -evaluate_centers(x),
        x0,
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 200}
    )

    optimal_centers = res_opt.x.reshape(-1, 2)

    # 4. Compute final radii with the optimized centers
    final_sum_radii = evaluate_centers(res_opt.x)
    centers_flat = res_opt.x
    
    # Re-run LP to extract radii
    centers = centers_flat.reshape(-1, 2)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        diff = centers[i, np.newaxis, :] - centers[np.newaxis, i, :]
        dist_matrix[i, i:] = np.sqrt(np.sum(diff**2, axis=1))

    c_obj = -np.ones(n)
    constraints_rows = []
    constraints_data = []
    constraints_b = []

    for i in range(n):
        x, y = centers[i]
        constraints_rows.extend([i]); constraints_data.extend([1.0]); constraints_b.append(x)
        constraints_rows.extend([i]); constraints_data.extend([1.0]); constraints_b.append(1.0 - x)
        constraints_rows.extend([i]); constraints_data.extend([1.0]); constraints_b.append(y)
        constraints_rows.extend([i]); constraints_data.extend([1.0]); constraints_b.append(1.0 - y)

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            constraints_rows.extend([i, j])
            constraints_data.extend([1.0, 1.0])
            constraints_b.append(d)

    rows = np.array(constraints_rows)
    cols = np.array(constraints_rows)
    A_ub = sp.csr_matrix((constraints_data, (rows, cols)), shape=(len(constraints_b), n))
    b_ub = np.array(constraints_b)

    lp_res = scipy.optimize.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, 
                                    bounds=[(0, None)] * n, method='highs')
    
    radii = lp_res.x

    return centers, radii, final_sum_radii
