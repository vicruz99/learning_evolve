# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d0344893) state=43cd724a sum of radii=2.551806 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Initialization ---
    # We start with a valid packing to give the optimizer a head start.
    # A 5x5 grid of circles with radius 0.09 fits easily (diameter 0.18, 5*0.18 = 0.9 < 1).
    # This gives 25 circles. We need 26.
    # We place the 26th circle in a gap or just add it with small radius.
    
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)
    
    idx = 0
    # Place 25 circles in a grid
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.18, 0.1 + j * 0.18]
            idx += 1
            
    # Place 26th circle. 
    # Let's put it in a gap. Center of square gap between (0.1, 0.1) and (0.28, 0.28) approx.
    # Actually, simple placement: near a corner or edge with small radius.
    # Or just random valid position.
    # Let's place it at (0.5, 0.5) with very small radius initially, 
    # but (0.5, 0.5) is crowded.
    # Let's place it at (0.05, 0.05) with radius 0.04. 
    # Distance to (0.1, 0.1) is sqrt(0.05^2 + 0.05^2) ~ 0.0707. 
    # Sum of radii 0.09 + 0.04 = 0.13 > 0.07. Overlap.
    
    # Better: Scale down all radii initially to ensure no overlap, then let optimizer grow them.
    # If we scale grid circles to r=0.05, they fit.
    # 26th circle can be small.
    
    # Reset for safe initialization
    radii = np.full(n, 0.05)
    centers = np.zeros((n, 2))
    
    # Fill 26 positions in a grid-like pattern but scaled to fit
    # We can fit roughly sqrt(26) ~ 5.1 items per side.
    # Let's use a 6x5 grid layout conceptually, but only fill 26 spots.
    # Spacing: (1 - 2*0.05) / 5 = 0.9 / 5 = 0.18.
    
    step = 0.18
    offset = 0.05
    
    idx = 0
    for r_idx in range(5): # 5 rows
        for c_idx in range(6): # 6 cols
            if idx < n:
                # Offset rows to create a slightly hexagonal packing hint?
                # Just straight grid for now.
                x = offset + c_idx * step
                y = offset + r_idx * step
                centers[idx] = [x, y]
                idx += 1
            else:
                break
        if idx >= n: break

    # Flatten variables for scipy: [x1, y1, r1, x2, y2, r2, ...]
    # Actually standard is all x, all y, all r? Or interleaved.
    # Let's use interleaved: [x1, y1, r1, x2, y2, r2, ...]
    # This makes indexing easier.
    # Variable vector length: 3 * n = 78
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Constraints
    cons = []
    
    # 1. Boundary Constraints:
    # x - r >= 0  => x - r >= 0
    # x + r <= 1  => 1 - x - r >= 0
    # y - r >= 0
    # y + r <= 1
    
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # 1 - x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
        })
        # y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # 1 - y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
        })

    # 2. Non-overlap Constraints:
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 3 * i
            idx_yi = 3 * i + 1
            idx_ri = 3 * i + 2
            
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            def overlap_constraint(v, i=i, j=j, 
                                   idx_xi=idx_xi, idx_yi=idx_yi, idx_ri=idx_ri,
                                   idx_xj=idx_xj, idx_yj=idx_yj, idx_rj=idx_rj):
                xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                return dist_sq - sum_r*sum_r

            cons.append({
                'type': 'ineq',
                'fun': overlap_constraint
            })

    # Objective function: Maximize sum of radii => Minimize negative sum
    def objective(v):
        # Sum of radii is sum of v[2], v[5], v[8]...
        # Slice: v[2::3]
        return -np.sum(v[2::3])

    # Optimization
    # Method SLSQP is suitable for constrained optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Extract results
    final_x = res.x
    
    centers_final = np.zeros((n, 2))
    radii_final = np.zeros(n)
    
    for i in range(n):
        centers_final[i, 0] = final_x[3*i]
        centers_final[i, 1] = final_x[3*i+1]
        radii_final[i] = final_x[3*i+2]
        
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii

# Ensure no closures/lambdas capture changing variables incorrectly if called multiple times
# The function above creates new functions inside, which is fine for a single run.
# But per instructions: "Make all helper functions top level and have no closures from function nesting."
# This implies I should move constraint functions out or define them without capturing loop variables via closure in a way that's unsafe?
# Actually, standard lambda in list comprehension captures loop variable by reference.
# In the code above, I used default arguments `i=i` in lambda/closure to fix the value. This is safe.
# However, the instruction "No closures from function nesting" might mean I shouldn't define functions inside run_packing.
# I will refactor to be safe.

def constraint_x_minus_r(v, i, n):
    return v[3*i] - v[3*i+2]

def constraint_one_minus_x_minus_r(v, i, n):
    return 1.0 - v[3*i] - v[3*i+2]

def constraint_y_minus_r(v, i, n):
    return v[3*i+1] - v[3*i+2]

def constraint_one_minus_y_minus_r(v, i, n):
    return 1.0 - v[3*i+1] - v[3*i+2]

def constraint_overlap(v, i, j):
    idx_xi = 3 * i
    idx_yi = 3 * i + 1
    idx_ri = 3 * i + 2
    
    idx_xj = 3 * j
    idx_yj = 3 * j + 1
    idx_rj = 3 * j + 2
    
    xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
    xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
    
    dx = xi - xj
    dy = yi - yj
    dist_sq = dx*dx + dy*dy
    sum_r = ri + rj
    return dist_sq - sum_r*sum_r

def objective_func(v):
    return -np.sum(v[2::3])

def run_packing_refactored() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Initialization
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05)
    
    step = 0.18
    offset = 0.05
    idx = 0
    for r_idx in range(5):
        for c_idx in range(6):
            if idx < n:
                centers[idx] = [offset + c_idx * step, offset + r_idx * step]
                idx += 1
            else:
                break
        if idx >= n: break

    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    cons = []
    for i in range(n):
        # Boundary constraints
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_x_minus_r(v, i, n)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_one_minus_x_minus_r(v, i, n)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_y_minus_r(v, i, n)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_one_minus_y_minus_r(v, i, n)})

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: constraint_overlap(v, i, j)})

    res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 2000, 'ftol': 1e-10})
    
    final_x = res.x
    
    centers_final = np.zeros((n, 2))
    radii_final = np.zeros(n)
    
    for i in range(n):
        centers_final[i, 0] = final_x[3*i]
        centers_final[i, 1] = final_x[3*i+1]
        radii_final[i] = final_x[3*i+2]
        
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii

# Replace run_packing with the refactored one
run_packing = run_packing_refactored
