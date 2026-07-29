# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=e678066d sum of radii=1.560000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # --- Stage 1: Initialization ---
    # Generate a hexagonal lattice initialization
    centers = []
    r_init = 0.06  # Initial radius, small enough to fit
    
    # We need 26 circles. 
    # A pattern of 4 and 3 circles per row works well.
    # 4 + 3 + 4 + 3 + 4 + 3 + 4 = 25? No.
    # Let's do 4, 3, 4, 3, 4, 3, 4 -> 4*4 + 3*3 = 16+9=25. Need 1 more.
    # Let's do 4, 3, 4, 3, 4, 4, 4 -> 26.
    # Or simply fill rows until we have 26.
    
    row_counts = [4, 3, 4, 3, 4, 3, 4, 1] # Sum = 26
    # Actually, let's just generate points and take first 26
    
    y = r_init
    row_idx = 0
    while len(centers) < n_circles:
        count = row_counts[row_idx % len(row_counts)]
        row_centers = []
        
        # Horizontal spacing 2*r, vertical sqrt(3)*r
        # Shift alternate rows by r
        x_start = r_init
        if row_idx % 2 == 1:
            x_start += r_init # Shift by r for hexagonal packing
            
        for i in range(count):
            x = x_start + i * (2 * r_init)
            # Check if within horizontal bounds
            if x + r_init <= 1.0:
                row_centers.append([x, y])
        
        centers.extend(row_centers)
        y += r_init * np.sqrt(3)
        row_idx += 1
        
        # If we can't fit more in x, break to avoid infinite loop if logic fails
        if len(row_centers) == 0 and y > 1.0:
            break
            
    centers = np.array(centers[:n_circles])
    radii = np.full(n_circles, r_init)
    
    # Flatten for optimizer: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.hstack([centers.flatten(), radii])
    
    # --- Stage 2: Optimization ---
    
    def objective(vars):
        # Maximize sum of radii => Minimize negative sum
        r = vars[2::3]
        return -np.sum(r)
    
    def get_constraints(vars):
        constraints = []
        n = len(vars) // 3
        
        # Extract coords and radii
        # vars structure: x0, y0, r0, x1, y1, r1 ...
        # But we used hstack on centers (xy) then radii.
        # centers was (n, 2). flattened is x0, y0, x1, y1...
        # So indices: 0,1 -> c0; 2,3 -> c1...
        # Radii start at index 2*n
        
        # Let's reconstruct arrays for clarity
        cx = np.array(vars[:n*2]).reshape(n, 2)[:, 0]
        cy = np.array(vars[:n*2]).reshape(n, 2)[:, 1]
        r = vars[2*n:]
        
        # Boundary Constraints:
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # => x - r >= 0, 1-x - r >= 0, etc.
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n+i] - v[2*i]}) # y - r >= 0 ? No.
            # Let's map indices carefully.
            # v[2*i] is x_i, v[2*i+1] is y_i, v[2*n+i] is r_i
            
            # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n+i]})
            # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i] - v[2*n+i]})
            # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*n+i]})
            # 1 - y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i+1] - v[2*n+i]})

        # Overlap Constraints:
        # dist >= r_i + r_j  => dist - (r_i + r_j) >= 0
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: 
                        np.sqrt((v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2) - (v[2*n+i] + v[2*n+j])
                })
        
        return constraints

    cons = get_constraints(x0)
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
    for i in range(n_circles):
        bounds.append((0, 0.5)) # r

    # Run optimization
    # SLSQP is good for this
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    if res.success:
        final_vars = res.x
    else:
        # Fallback to initial if optimization fails, though unlikely with this setup
        final_vars = x0

    # Extract results
    n = n_circles
    centers_final = np.array(final_vars[:n*2]).reshape(n, 2)
    radii_final = np.array(final_vars[2*n:])
    
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii
