# sol_000362 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b4024b4) state=58cef0dd sum of radii=1.412611 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def solve_radii(centers):
    """
    Solve for the optimal radii for a fixed set of centers using Linear Programming.
    """
    n = centers.shape[0]
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Precompute distances for pair-wise constraints
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e12) # Avoid self-interaction in lower triangle

    # Pairwise constraints: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1
        
        A_ub.append(row); b_ub.append(x)
        A_ub.append(row); b_ub.append(1 - x)
        A_ub.append(row); b_ub.append(y)
        A_ub.append(row); b_ub.append(1 - y)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return -res.fun, res.x
    return 0, np.zeros(n)

def loss_function(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    # Clip to valid range to ensure LP doesn't fail on boundary logic
    centers = np.clip(centers, 1e-6, 1 - 1e-6)
    val, _ = solve_radii(centers)
    return -val

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # 1. Hexagonal-like Grid Initialization
    init_centers = []
    row_count = 0
    x_start = 0.1
    while row_count < n:
        y_pos = row_count * 0.18 + 0.1
        if y_pos > 0.95: break
        
        if row_count % 2 == 0:
            cols = 5
            x_off = x_start
        else:
            cols = 6
            x_off = x_start + 0.09
            
        x_positions = [x_off + i * 0.18 for i in range(cols)]
        for x in x_positions:
            if row_count < n:
                init_centers.append([x, y_pos])
                row_count += 1
            else:
                break
                
    # Fill remaining if any (though 5x5 + shifts usually cover 26)
    while len(init_centers) < n:
        init_centers.append([np.random.rand(), np.random.rand()])
        
    best_centers = np.array(init_centers[:n])
    best_sum, best_radii = solve_radii(best_centers)
    
    # 2. Optimization with Nelder-Mead
    # We run a few restarts to avoid local minima
    np.random.seed(42)
    for restart in range(3):
        if restart > 0:
            centers_to_opt = best_centers + np.random.randn(26, 2) * 0.05
            centers_to_opt = np.clip(centers_to_opt, 0.01, 0.99)
        else:
            centers_to_opt = best_centers.copy()
            
        res = minimize(loss_function, centers_to_opt.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.success:
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = res.x.reshape(-1, 2)
                # Re-solve radii for clean result
                _, best_radii = solve_radii(best_centers)

    return best_centers, best_radii, best_sum
