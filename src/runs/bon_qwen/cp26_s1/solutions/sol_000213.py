# sol_000213 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dddb8969) state=5adde2c0 sum of radii=2.231765 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """
    Given fixed centers, solve the LP to find radii that maximize the sum
    while respecting non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    c = -np.ones(n)  # Maximize sum -> minimize negative sum
    A_ub_list = []
    b_ub_list = []

    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1.0
        
        A_ub_list.append(row.copy()); b_ub_list.append(x)
        A_ub_list.append(row.copy()); b_ub_list.append(1.0 - x)
        A_ub_list.append(row.copy()); b_ub_list.append(y)
        A_ub_list.append(row.copy()); b_ub_list.append(1.0 - y)

    # Non-overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(dist)

    A_ub = np.vstack(A_ub_list)
    b_ub = np.array(b_ub_list)
    bounds = [(0.0, 0.5) for _ in range(n)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'disp': False})
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    # Fallback in case of LP failure
    return np.full(n, 0.01), 0.01 * n

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal grid pattern
    centers = np.zeros((n, 2))
    idx = 0
    # Parameters for hex packing
    base_r = 0.1
    y_step = base_r * np.sqrt(3)
    
    for r_idx in range(8):
        y = r_idx * y_step + base_r
        for c_idx in range(8):
            x = c_idx * 2 * base_r + (r_idx % 2) * base_r + base_r
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
        if idx >= n:
            break
            
    # Normalize to unit square with margin
    centers -= centers.min(axis=0)
    span = centers.max(axis=0) - centers.min(axis=0)
    centers /= span * 1.1
    centers = np.clip(centers, 0.05, 0.95)
    
    # Add small random perturbation to break symmetry
    centers += np.random.normal(0, 0.005, size=(n, 2))
    centers = np.clip(centers, 0.02, 0.98)

    # 2. Simulated Annealing
    best_centers = centers.copy()
    best_radii, best_sum = get_optimal_radii(centers)
    curr_centers = centers.copy()
    curr_sum = best_sum
    
    temp = 0.04
    step_size = 0.03
    iterations = 3500
    
    for step in range(iterations):
        # Propose new centers
        move = np.random.normal(0, step_size, size=(n, 2))
        proposed = np.clip(curr_centers + move, 0.01, 0.99)
        
        # Evaluate
        prop_radii, prop_sum = get_optimal_radii(proposed)
        
        # SA acceptance criterion
        delta = prop_sum - curr_sum
        if delta > 0:
            curr_centers = proposed
            curr_sum = prop_sum
            if prop_sum > best_sum:
                best_sum = prop_sum
                best_centers = proposed.copy()
                best_radii = prop_radii.copy()
        else:
            if np.random.rand() < np.exp(delta / max(temp, 1e-9)):
                curr_centers = proposed
                curr_sum = prop_sum
                
        # Cooling schedule
        temp *= 0.9993
        step_size *= 0.9995
        
    return best_centers, best_radii, float(best_sum)
