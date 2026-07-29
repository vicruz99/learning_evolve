# sol_000235 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dc099519) state=fe16bb55 sum of radii=2.592613 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns the best valid packing of 26 circles found by optimization.
    """
    n = 26
    best_result = None
    best_score = -1.0
    
    # Try multiple restarts to find the global optimum
    for seed in range(10):
        centers, radii = get_initial_guess(n, seed)
        
        # Combine into a single optimization vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[i*3]   = centers[i, 0]
            x0[i*3+1] = centers[i, 1]
            x0[i*3+2] = radii[i]
        
        # Constraints
        cons = []
        
        # Boundary and positivity constraints
        for i in range(n):
            # x >= r, x <= 1-r, y >= r, y <= 1-r, r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})          # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i*3] - x[i*3+2]})    # x + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})        # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i*3+1] - x[i*3+2]})  # y + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2]})                   # r >= 0
            
        # Non-overlap constraints: dist >= r1 + r2
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j)})

        # Objective: maximize sum of radii => minimize negative sum
        res = minimize(
            fun=objective_function,
            x0=x0,
            method='SLSQP',
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        current_score = -res.fun
        if current_score > best_score:
            best_score = current_score
            best_result = res.x
            
    # Format the best result
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_result[i*3]
        centers[i, 1] = best_result[i*3+1]
        radii[i] = best_result[i*3+2]
        
    return centers, radii, best_score

def objective_function(x):
    """Minimize the negative sum of radii."""
    n = len(x) // 3
    total_r = 0.0
    for i in range(n):
        total_r += x[i*3+2]
    return -total_r

def overlap_constraint(x, i, j):
    """Distance between centers >= sum of radii."""
    x1, y1, r1 = x[i*3], x[i*3+1], x[i*3+2]
    x2, y2, r2 = x[j*3], x[j*3+1], x[j*3+2]
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist - (r1 + r2)

def get_initial_guess(n, seed):
    """Generates a hybrid initial guess: large corner/edge circles, smaller central ones."""
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    idx = 0
    
    # 1. 4 Corners (Large)
    for x, y in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        centers[idx] = [x, y]
        radii[idx] = 0.15
        idx += 1
        
    # 2. 6 Edges (Medium)
    edge_positions = [
        (0.5, 0.1), (0.1, 0.5), (0.9, 0.5), 
        (0.5, 0.9), (0.25, 0.08), (0.75, 0.08)
    ]
    for x, y in edge_positions:
        centers[idx] = [x, y]
        radii[idx] = 0.10
        idx += 1
        
    # 3. Remaining 16 (Central)
    while idx < n:
        # Random placement with some jitter around a grid
        cx = 0.2 + 0.6 * np.random.rand()
        cy = 0.2 + 0.6 * np.random.rand()
        centers[idx] = [cx, cy]
        radii[idx] = 0.05 + 0.05 * np.random.rand()
        idx += 1
        
    return centers, radii
