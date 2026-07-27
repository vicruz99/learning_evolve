import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def run_packing():
    n = 26
    # 1. Initialization: Hexagonal layout (5 and 6 per row)
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        k = 6 if row % 2 == 0 else 5
        y = 0.05 + row * 0.19
        if row % 2 == 1:
            y += 0.09
        for i in range(k):
            if idx < n:
                x = 0.05 + i * 0.18 + (row % 2) * 0.09
                centers[idx] = [x, y]
                idx += 1
    
    radii = np.full(n, 0.10)

    # 2. Optimization Function
    def objective(vars):
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:]
        return -np.sum(r) # Negative for minimization

    def boundary_constraints(vars):
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:]
        c = []
        for i in range(n):
            c.append(x[i] - r[i])
            c.append(1.0 - x[i] - r[i])
            c.append(y[i] - r[i])
            c.append(1.0 - y[i] - r[i])
        return c

    def overlap_constraints(vars):
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:]
        c = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx**2 + dy**2
                min_dist = r[i] + r[j] + 1e-5 # Small tolerance
                c.append(dist_sq - min_dist**2)
        return c

    # 3. Setup and Execute Optimization
    x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    
    cons = (
        {"type": "ineq", "fun": boundary_constraints},
        {"type": "ineq", "fun": overlap_constraints}
    )
    
    bounds = [
        (0.0, 1.0) for _ in range(2 * n)
    ] + [(1e-4, 0.5) for _ in range(n)]

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 2000, 'ftol': 1e-10})

    # 4. Post-process and Clean
    final_x = result.x[:n]
    final_y = result.x[n:2*n]
    final_r = result.x[2*n:]

    # Ensure strict validity
    final_centers = np.column_stack([final_x, final_y])
    
    # Final radius clamping for absolute safety
    for i in range(n):
        max_r = min(final_centers[i, 0], 1 - final_centers[i, 0],
                    final_centers[i, 1], 1 - final_centers[i, 1])
        final_r[i] = min(final_r[i], max_r)

    # Validate and adjust for overlaps
    valid = False
    attempts = 0
    while not valid and attempts < 5:
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(final_centers[i] - final_centers[j])
                if d < final_r[i] + final_r[j] - 1e-12:
                    factor = d / (final_r[i] + final_r[j] + 1e-9)
                    final_r[i] *= factor
                    final_r[j] *= factor
        valid = validate_packing(final_centers, final_r)
        attempts += 1

    return final_centers, final_r, np.sum(final_r)