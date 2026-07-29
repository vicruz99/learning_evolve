# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3353d097) state=5f5ebd5e sum of radii=2.583585 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns a packing of 26 circles in a unit square maximizing the sum of radii.
    Uses an optimized hybrid hexagonal packing structure.
    """
    
    def compute_objective(params):
        # params: [r1, r2, r3, r4, r5, center offsets...]
        # We will use a simpler formulation: fixed layout, optimize radii.
        pass

    # 1. Define the optimal layout structure for 26 circles
    # Radii for the five rows
    r1, r2, r3, r4, r5 = 0.0906, 0.1052, 0.1000, 0.1052, 0.0850
    
    # 2. Initialize centers based on hexagonal symmetry
    centers = []
    
    # Row 1: 6 circles
    for i in range(6):
        x = 0.5 + (i - 2.5) * (2 * r1)
        centers.append([x, r1])
        
    # Row 2: 5 circles
    for i in range(5):
        x = 0.5 + (i - 2) * (2 * r2)
        centers.append([x, r1 + np.sqrt(3) * r2])
        
    # Row 3: 5 circles
    for i in range(5):
        x = 0.5 + (i - 2) * (2 * r3)
        centers.append([x, r1 + np.sqrt(3) * r2 + np.sqrt(3) * r3])
        
    # Row 4: 6 circles
    for i in range(6):
        x = 0.5 + (i - 2.5) * (2 * r4)
        centers.append([x, r1 + np.sqrt(3) * r2 + np.sqrt(3) * r3 + np.sqrt(3) * r4])
        
    # Row 5: 4 circles
    for i in range(4):
        x = 0.5 + (i - 1.5) * (2 * r5)
        centers.append([x, r1 + np.sqrt(3) * r2 + np.sqrt(3) * r3 + np.sqrt(3) * r4 + np.sqrt(3) * r5])
        
    centers = np.array(centers)
    radii = np.array([r1]*6 + [r2]*5 + [r3]*5 + [r4]*6 + [r5]*4)
    
    # 3. Numerical Optimization to maximize sum of radii
    # Variables: x_coords (26), y_coords (26), radii (26)
    x0 = np.hstack([centers[:, 0], centers[:, 1], radii])
    n = 26

    def objective(x_all):
        # Minimize the negative sum of radii
        radii_part = x_all[2*n:]
        return -np.sum(radii_part)

    def constraints(x_all):
        x = x_all[:n]
        y = x_all[n:2*n]
        r = x_all[2*n:]
        cons = []
        
        # Boundary constraints: x, y, 1-x, 1-y >= r
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x_all, i=i: x_all[i] - x_all[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda x_all, i=i: x_all[n+i] - x_all[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda x_all, i=i: 1 - x_all[i] - x_all[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda x_all, i=i: 1 - x_all[n+i] - x_all[2*n+i]})
            
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x_all, i=i, j=j: \
                    (x_all[i] - x_all[j])**2 + (x_all[n+i] - x_all[n+j])**2 - (x_all[2*n+i] + x_all[2*n+j])**2
                })
        return cons

    # Optimize
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        constraints=constraints(x0),
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # Final extraction
    best_centers = np.column_stack((result.x[:n], result.x[n:2*n]))
    best_radii = result.x[2*n:]
    
    # Ensure radii are non-negative
    best_radii = np.maximum(best_radii, 1e-9)
    
    # Final validation check (internal)
    # Adjust centers to ensure strict boundary compliance if numerical errors occur
    for i in range(n):
        best_radii[i] = min(best_radii[i], best_centers[i,0], best_centers[i,1], 1-best_centers[i,0], 1-best_centers[i,1])
        best_radii[i] = max(best_radii[i], 1e-9)

    return best_centers, best_radii, np.sum(best_radii)
