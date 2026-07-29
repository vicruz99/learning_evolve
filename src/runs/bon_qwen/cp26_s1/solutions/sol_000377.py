# sol_000377 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7a0a6c4a) state=a182a677 sum of radii=2.583675 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def solve_radii(centers):
    """
    Given a set of centers, solve for the optimal radii that maximize 
    the sum of radii subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    c = -np.ones(n)  # Maximize sum of radii (minimize negative sum)
    
    A = []
    b = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    for i in range(n):
        x, y = centers[i]
        # r_i <= x_i  => r_i <= x_i
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(x)
        
        # r_i <= 1 - x_i => r_i <= 1-x_i
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(1 - x)
        
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(y)
        
        # r_i <= 1 - y_i
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(1 - y)
        
    # Non-overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A.append(row)
            b.append(dist)
            
    A = np.array(A)
    b = np.array(b)
    
    # Non-negative radii
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.fun * -1, res.x
    else:
        # Fallback if LP fails
        return 0, np.zeros(n)

def get_sum_radii(centers_flat):
    """Objective function for scipy minimize: returns negative sum of radii."""
    centers = centers_flat.reshape(-1, 2)
    sum_r, _ = solve_radii(centers)
    return -sum_r

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the packing algorithm to maximize sum of radii for 26 circles.
    """
    n = 26
    
    # --- 1. Initialize Centers (Hexagonal Pattern) ---
    centers = []
    rows = 6
    cols = 5 # Total 30 spots, we'll use 26
    r_init = 0.09
    
    # Generate hex lattice
    y_coord = r_init
    for i in range(rows):
        x_coord = r_init + (i % 2) * r_init
        for j in range(cols):
            if len(centers) >= n:
                break
            centers.append([x_coord, y_coord])
            x_coord += 2 * r_init
        y_coord += r_init * np.sqrt(3)
        
    centers = np.array(centers[:n])
    
    # --- 2. Optimize Centers ---
    # Flatten centers for optimizer
    x0 = centers.flatten()
    
    # Bounds for centers: [0, 1] for x and y
    bounds = [(0, 1)] * (2 * n)
    
    # Use SLSQP to find better positions
    # Nelder-Mead or SLSQP are good choices. SLSQP handles bounds well.
    res = minimize(get_sum_radii, x0, method='SLSQP', bounds=bounds, 
                   options={'maxiter': 100, 'ftol': 1e-6})
    
    best_centers = res.x.reshape(-1, 2)
    
    # --- 3. Final Radius Calculation ---
    sum_radii, radii = solve_radii(best_centers)
    
    # Ensure constraints are met strictly for the final output
    # (The LP ensures them, but numerical errors might exist)
    # The validate function allows 1e-12 tolerance.
    
    return best_centers, radii, sum_radii
