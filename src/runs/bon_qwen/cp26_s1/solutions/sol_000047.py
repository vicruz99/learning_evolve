# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=81076af0 sum of radii=2.607501 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Objective: Minimize negative sum of radii (equivalent to maximizing sum)
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)

    # Constraints:
    # 1. Boundary constraints (x, y, r within [0,1] and circle inside square)
    # 2. Non-overlap constraints (distance >= sum of radii)
    def constraints(x):
        cons = []
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        for i in range(n):
            centers[i, 0] = x[3*i]
            centers[i, 1] = x[3*i + 1]
            radii[i] = x[3*i + 2]
            
            # Boundary constraints: circle must be inside [0,1]x[0,1]
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i + 2]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i + 2]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i + 1] - x[3*i + 2]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i + 1] - x[3*i + 2]})
            # r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i + 2]})

        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                def overlap_constraint(x, i=i, j=j):
                    dist = np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2)
                    r_sum = x[3*i+2] + x[3*j+2]
                    return dist - r_sum
                cons.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return cons

    # Initialization: 5x5 grid + 1 circle in the center gap
    x0 = np.zeros(3 * n)
    # 5x5 Grid for first 25 circles
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            x0[3*idx] = 0.1 + j * 0.2
            x0[3*idx+1] = 0.1 + i * 0.2
            x0[3*idx+2] = 0.1
            
    # 26th circle in the center
    x0[3*25] = 0.5
    x0[3*25+1] = 0.5
    x0[3*25+2] = 0.04 # Small initial radius

    # Optimization bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Solve
    constraints_list = constraints(x0)
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints_list, 
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    # Extract results
    best_x = result.x
    centers = np.array([[best_x[3*i], best_x[3*i+1]] for i in range(n)])
    radii = np.array([best_x[3*i+2] for i in range(n)])
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
