import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initialization ---
    # Generate a hexagonal grid to provide a dense, structured initial guess
    centers = []
    spacing = 0.20  # Initial spacing estimate
    y = spacing
    row = 0
    while y <= 1 - spacing and len(centers) < n:
        x = spacing
        offset = spacing / 2 if row % 2 == 1 else 0
        while x + offset <= 1 - spacing and len(centers) < n:
            centers.append([x + offset, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    
    centers = np.array(centers)
    radii = np.full(n, 0.05)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.hstack([centers.flatten(), radii])
    
    # --- 2. Optimization Setup ---
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1) for _ in range(2 * n)] + [(0, 0.5) for _ in range(n)]
    
    constraints = []
    
    # Boundary Constraints: x_i >= r_i, 1 - x_i >= r_i, etc.
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[idx_x] - v[idx_r]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[idx_y] - v[idx_r]
        })

    # Overlap Constraints: dist(i, j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3 * i, 3 * i + 1, 3 * i + 2
            idx_xj, idx_yj, idx_rj = 3 * j, 3 * j + 1, 3 * j + 2
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) - (v[3*i+2] + v[3*j+2])
            })

    # Objective: Minimize -sum(radii)
    def objective(v):
        radii = v[2::3]
        return -np.sum(radii)

    # --- 3. Execution ---
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-9})

    # --- 4. Result Extraction ---
    final_vars = result.x
    final_centers = np.array([[final_vars[3*i], final_vars[3*i+1]] for i in range(n)])
    final_radii = final_vars[2::3]
    total_sum = np.sum(final_radii)

    return final_centers, final_radii, total_sum