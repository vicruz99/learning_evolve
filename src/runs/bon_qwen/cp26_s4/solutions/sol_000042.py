# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 12653929) state=085d0d52 sum of radii=2.490803 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def get_bounds():
    """Define variable bounds for the optimization variables [x1, y1, r1, ..., x26, y26, r26]"""
    bounds = []
    for _ in range(26):
        # x, y bounds: [0, 1]
        # r bounds: [0.001, 0.5] (r must be positive)
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.001, 0.5))
    return bounds

def get_initial_guess():
    """Generate a stable initial guess using a relaxed grid layout"""
    centers = []
    # 5x5 grid with some margin to avoid boundary conflicts
    for row in range(5):
        for col in range(5):
            centers.append([0.1 + col * 0.2, 0.1 + row * 0.2])
    # Add the 26th circle in the center
    centers.append([0.5, 0.5])
    
    # Initial radii
    radii = [0.05] * 26
    
    # Concatenate into a single vector [x1, y1, r1, x2, y2, r2, ...]
    guess = []
    for c, r in zip(centers, radii):
        guess.extend([c[0], c[1], r])
    return np.array(guess)

def constraint_func(vars):
    """
    Compute constraint violations for all pairs and boundaries.
    Constraints are returned as a dictionary for SLSQP.
    """
    n = 26
    vars = np.array(vars)
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i] = [vars[3*i], vars[3*i+1]]
        radii[i] = vars[3*i+2]
        
    violations = []
    
    # Boundary constraints: x - r >= 0, x + r <= 1, etc.
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        violations.append(x - r) # >= 0
        violations.append(y - r) # >= 0
        violations.append(1 - x - r) # >= 0
        violations.append(1 - y - r) # >= 0
        
    # Overlap constraints: dist(i,j) - (r_i + r_j) >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            violations.append(dist - (radii[i] + radii[j]))
            
    return violations

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Setup Bounds and Initial Guess
    bounds = get_bounds()
    x0 = get_initial_guess()
    
    # 2. Define the Optimization Problem
    # Objective: Maximize sum of radii (Minimize negative sum)
    def objective(vars):
        return -sum(vars[3*i+2] for i in range(26))
        
    # Constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # 3. Optimization
    # Use SLSQP to find the local maximum sum of radii
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    # 4. Extract Results
    final_vars = result.x
    centers = np.zeros((26, 2))
    radii = np.zeros(26)
    
    for i in range(26):
        centers[i] = [final_vars[3*i], final_vars[3*i+1]]
        radii[i] = final_vars[3*i+2]
        
    sum_radii = np.sum(radii)
    
    # 5. Safety: Clamp tiny negative radii to 0 (though bounds prevent this)
    radii = np.clip(radii, 0, None)
    
    return centers, radii, sum_radii
