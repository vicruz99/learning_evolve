import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialization
    # Distribute centers in a 6x5 grid pattern to start with a dense packing
    centers = np.zeros((n, 2))
    for i in range(n):
        centers[i, 0] = 0.1 + (i % 6) * 0.16
        centers[i, 1] = 0.1 + (i // 6) * 0.25
        
    radii = np.full(n, 0.08)
    
    # 2. Objective function (with penalty method)
    penalty_weight = 1000.0
    
    def objective(x):
        c = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        
        # 1. Sum of radii (to maximize)
        obj = -np.sum(r)
        
        # 2. Penalty for boundary violations
        # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        violations = np.maximum(0, r - c[:, 0])
        violations += np.maximum(0, r - c[:, 1])
        violations += np.maximum(0, c[:, 0] + r - 1)
        violations += np.maximum(0, c[:, 1] + r - 1)
        obj += penalty_weight * np.sum(violations ** 2)
        
        # 3. Penalty for overlaps
        # distance_ij - r_i - r_j >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                overlap = (r[i] + r[j]) - dist
                if overlap > 0:
                    obj += penalty_weight * overlap ** 2
                    
        return obj

    # 3. Optimization
    # Bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    x0 = np.concatenate([centers.flatten(), radii])

    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 2000}
    )

    # 4. Post-processing and cleanup
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # Clamp small values to ensure strict non-negative and boundary constraints
    final_radii = np.maximum(final_radii, 0.0)
    final_radii = np.minimum(final_radii, np.minimum(
        np.minimum(final_centers[:, 0], 1 - final_centers[:, 0]),
        np.minimum(final_centers[:, 1], 1 - final_centers[:, 1])
    ))
    
    # Recalculate sum
    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")