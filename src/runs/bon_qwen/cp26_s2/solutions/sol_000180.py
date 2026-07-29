# sol_000180 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0ae2e142) state=037351c3 sum of radii=2.627565 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal-like grid
    # We want to place 26 circles. 
    # A 5x5 grid has 25. We can place the 26th in a gap or just perturb.
    # Let's try a staggered grid (hexagonal) which is denser.
    
    centers = np.zeros((n, 2))
    radii_init = np.full(n, 0.08) # Start with a feasible small radius
    
    # Place circles in a staggered pattern
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle (centered)
    
    idx = 0
    rows = 5
    cols = 5
    # Spacing for 5 circles in width 1 is 0.2. 
    # Hexagonal vertical spacing is 0.2 * sqrt(3)/2 approx 0.1732
    # Total height for 5 rows approx 4*0.1732 + 2*radius.
    
    x_step = 1.0 / 6.0 # Slightly less than 0.2 to leave margin? No, 1/5 = 0.2.
    # Let's use 0.2 spacing
    x_step = 0.2
    
    # We need to fit 5 circles of width ~0.2.
    # Centers at 0.1, 0.3, 0.5, 0.7, 0.9 fits perfectly with r=0.1.
    # With r=0.08, it fits easily.
    
    for r in range(rows):
        # Shift odd rows by half step
        offset = (x_step / 2) * (r % 2)
        for c in range(cols):
            if idx < n:
                cx = 0.1 + c * x_step + offset
                cy = 0.1 + r * (x_step * np.sqrt(3) / 2)
                
                # Clamp to [0.05, 0.95] roughly to be safe initially
                cx = max(0.05, min(0.95, cx))
                cy = max(0.05, min(0.95, cy))
                
                centers[idx] = [cx, cy]
                idx += 1
    
    # If we didn't reach 26 (though 5x5=25, we need 1 more)
    # The loop above puts 25. The 26th one is missing.
    # Let's add the 26th circle in a likely gap, e.g., center if not occupied, 
    # or just random valid position.
    if idx < n:
        # Place remaining circles in available space
        # Try center
        centers[idx] = [0.5, 0.5]
        idx += 1
        while idx < n:
            centers[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
            idx += 1

    # 2. Setup Optimization Variables
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Size: 3 * n
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii_init[i]

    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1) for _ in range(3 * n)]
    # Tighten r bound to 0.5 (max possible)
    for i in range(n):
        bounds[3*i+2] = (0, 0.5)

    # 3. Constraints
    # Inequality constraints: g(x) >= 0
    cons = []

    # Boundary constraints
    # r <= x <= 1-r  =>  x - r >= 0  and  1 - x - r >= 0
    # r <= y <= 1-r  =>  y - r >= 0  and  1 - y - r >= 0
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})      # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})# 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})    # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})# 1 - y - r >= 0

    # Non-overlap constraints
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # 4. Objective Function
    # Maximize sum of radii -> Minimize -sum(r)
    def objective(vars):
        total_r = 0.0
        for i in range(n):
            total_r += vars[3*i+2]
        return -total_r

    # 5. Run Optimization
    # SLSQP is a good choice for this type of problem
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    
    # 6. Extract Solution
    best_vars = result.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_vars[3*i]
        final_centers[i, 1] = best_vars[3*i+1]
        final_radii[i] = best_vars[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    # 7. Post-processing / Safety check
    # Ensure strict feasibility if close to boundary
    # The optimizer might return values like 1.0000000001 due to float precision
    # But constraints should handle it. 
    # However, for the validation function, we should be safe.
    
    # Re-clamp radii if they are extremely negative (shouldn't happen)
    final_radii = np.maximum(final_radii, 0)
    
    # Adjust centers to be strictly inside if radius is large?
    # The constraints x-r >= 0 etc ensure this.
    
    return final_centers, final_radii, sum_radii
