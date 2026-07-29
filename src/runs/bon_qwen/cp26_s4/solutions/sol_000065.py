# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 10bf7585) state=f7c01cf0 sum of radii=1.170947 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the placement and radii of 26 circles in a unit square 
    to maximize the sum of radii.
    """
    N = 26
    n_circles = N

    # 1. Initialize Hexagonal Lattice
    centers = np.zeros((n_circles, 2))
    radii = np.full(n_circles, 0.09)
    
    # Pattern: 5, 6, 5, 6, 4 (Total 26 circles)
    row_counts = [5, 6, 5, 6, 4]
    idx = 0
    for i, count in enumerate(row_counts):
        # Horizontal spacing
        x_start = (1 - count * 2 * radii[0]) / 2 + radii[0]
        for j in range(count):
            centers[idx, 0] = x_start + j * 2 * radii[0]
            # Hexagonal vertical spacing with horizontal offset
            if i % 2 == 1: 
                centers[idx, 0] += radii[0]
            centers[idx, 1] = radii[0] + i * np.sqrt(3) * radii[0]
            idx += 1
            
    # Scale and normalize to fit in unit square initially
    centers[:, 0] = (centers[:, 0] - centers[:, 0].min()) * 0.8 + 0.1
    centers[:, 1] = (centers[:, 1] - centers[:, 1].min()) * 0.8 + 0.1

    # 2. Objective and Constraint Functions
    def objective(x):
        # x: (x1, y1, r1, x2, y2, r2, ...)
        return -np.sum(x[2::3]) # Maximize sum of radii

    def boundary_constraints(x):
        con = []
        for i in range(n_circles):
            r = x[2 + 3*i]
            x_pos = x[3*i]
            y_pos = x[3*i + 1]
            # x - r >= 0, 1 - (x + r) >= 0, etc.
            con.append(x_pos - r)
            con.append(1.0 - x_pos - r)
            con.append(y_pos - r)
            con.append(1.0 - y_pos - r)
            # Radius lower bound
            con.append(r - 0.001)
        return np.array(con)

    def overlap_constraints(x):
        con = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                xi, yi, ri = x[3*i], x[3*i + 1], x[3*i + 2]
                xj, yj, rj = x[3*j], x[3*j + 1], x[3*j + 2]
                # distance >= r1 + r2 -> distance - (r1 + r2) >= 0
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                con.append(dist - (ri + rj))
        return np.array(con)

    def combined_constraints(x):
        return np.concatenate([boundary_constraints(x), overlap_constraints(x)])

    # 3. Optimization Setup
    initial_x = np.zeros(n_circles * 3)
    for i in range(n_circles):
        initial_x[3*i] = centers[i, 0]
        initial_x[3*i + 1] = centers[i, 1]
        initial_x[3*i + 2] = radii[i]

    bounds = [(0.01, 0.5)] * (n_circles * 3)
    
    # Run optimization with a robust method
    # Using SLSQP as it handles non-linear constraints well
    try:
        res = minimize(objective, initial_x, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': combined_constraints},
                       options={'maxiter': 2000, 'ftol': 1e-9})
        optimal_x = res.x
    except Exception:
        optimal_x = initial_x

    # 4. Reconstruction and Final Refinement (Shrinking)
    final_centers = np.array([[optimal_x[3*i], optimal_x[3*i + 1]] for i in range(n_circles)])
    final_radii = np.array([optimal_x[3*i + 2] for i in range(n_circles)])

    # Iterative shrinking to guarantee no overlap
    for _ in range(5):
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(final_centers[i] - final_centers[j])
                req_sum = final_radii[i] + final_radii[j]
                if dist < req_sum:
                    overlap = req_sum - dist
                    reduction = overlap / 2 + 1e-9
                    final_radii[i] -= reduction
                    final_radii[j] -= reduction
        
        # Boundary shrinking
        for i in range(n_circles):
            x, y = final_centers[i]
            r = final_radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                r = min(x, 1 - x, y, 1 - y) - 1e-9
                final_radii[i] = max(r, 0.001)

    final_sum = np.sum(final_radii)
    return final_centers, final_radii, final_sum

# Validate the output
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.5f}")
