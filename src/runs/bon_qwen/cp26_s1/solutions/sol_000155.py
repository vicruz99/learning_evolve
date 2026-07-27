# sol_000155 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39a6f529) state=a7baac38 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    
    # 1. Initialize centers and radii
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.05) # Start with small feasible radii

    # Hexagonal-ish grid layout to maximize initial density
    row_counts = [5, 5, 5, 5, 5, 1]
    idx = 0
    current_y = 0.0
    
    for i, count in enumerate(row_counts):
        # Alternate x-offsets for hexagonal packing
        x_offset = 0.1 if i % 2 == 1 else 0.0
        step = 1.0 / (count + 1)
        
        for j in range(count):
            x = (j + 1) * step + x_offset
            y = current_y
            
            # Ensure initial positions are valid and inside
            centers[idx] = [np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)]
            idx += 1
        
        current_y += 0.15
        if idx >= N:
            break
            
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = np.zeros(3 * N)
    for i in range(N):
        initial_vars[3 * i] = centers[i, 0]
        initial_vars[3 * i + 1] = centers[i, 1]
        initial_vars[3 * i + 2] = radii[i]

    def objective(vars_vec):
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(vars_vec[2::3])

    def constraints(vars_vec):
        cons = []
        
        # Extract variables
        centers_opt = np.array([vars_vec[i:i+2] for i in range(0, 3*N, 3)])
        radii_opt = vars_vec[2::3]
        
        # Boundary constraints for each circle
        for i in range(N):
            x, y = centers_opt[i]
            r = radii_opt[i]
            
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]})
            # x + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx+2]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]})
            # y + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx+1] - v[3*idx+2]})
            # r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx+2]})
            
        # Pairwise non-overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                def dist_sq(v, i=i, j=j):
                    xi, yi = v[3*i], v[3*i+1]
                    xj, yj = v[3*j], v[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2
                
                def radii_sum(v, i=i, j=j):
                    return (v[3*i+2] + v[3*j+2])**2
                
                # dist^2 >= (r_i + r_j)^2
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: dist_sq(v, i, j) - radii_sum(v, i, j)})
                
        return cons

    # Run optimization
    # Bounds to keep variables somewhat reasonable during optimization
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        result = minimize(objective, initial_vars, method='SLSQP', 
                          bounds=bounds, 
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-9})
        
        final_vars = result.x
        centers_final = np.array([[final_vars[3*i], final_vars[3*i+1]] for i in range(N)])
        radii_final = final_vars[2::3]
        
    except Exception as e:
        # Fallback to the initial valid configuration if optimization fails
        centers_final = centers
        radii_final = radii

    # Post-processing: Clamp values to strictly satisfy [0,1] and non-negative radius
    # This handles potential floating point drift near boundaries
    for i in range(N):
        r = radii_final[i]
        x, y = centers_final[i]
        
        # Clamp radius to fit within box
        r = min(r, x, 1.0 - x, y, 1.0 - y, 0.5)
        radii_final[i] = max(r, 0.0)
        
        # Ensure center is safe distance from boundary
        centers_final[i, 0] = np.clip(x, radii_final[i], 1.0 - radii_final[i])
        centers_final[i, 1] = np.clip(y, radii_final[i], 1.0 - radii_final[i])

    total_sum = np.sum(radii_final)
    return centers_final, radii_final, float(total_sum)
