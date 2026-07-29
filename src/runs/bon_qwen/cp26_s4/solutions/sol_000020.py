# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d7881bb) state=1be070f6 sum of radii=2.616510 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Helper Functions ---

    def get_constraints_equal(centers, r_val):
        """
        Constraints for equal radius optimization.
        Returns list of constraint dictionaries.
        """
        constraints = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x_i >= r
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[-1]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[-1]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[-1]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i+1] - v[-1]})
            
        # Non-overlap constraints: dist >= 2r
        # dist^2 >= 4r^2
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: 
                        (v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2 - 4 * (v[-1])**2
                })
        
        return constraints

    def get_constraints_unequal(centers, radii):
        """
        Constraints for unequal radius optimization.
        Variables: [x1, y1, x2, y2, ..., r1, r2, ...]
        Actually, to keep vector simple: [x1, y1, ..., x26, y26, r1, ..., r26]
        """
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[n*2 + i]})
            # x <= 1 - r  => 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[n*2 + i]})
            # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[n*2 + i]})
            # y <= 1 - r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i+1] - v[n*2 + i]})
            
        # Non-overlap: dist >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j:
                        (v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2 - (v[n*2 + i] + v[n*2 + j])**2
                })
        
        # Radii non-negative
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n*2 + i]})

        return constraints

    def objective_equal(v):
        # Maximize r (last variable) => Minimize -r
        return -v[-1]

    def objective_unequal(v):
        # Maximize sum of radii => Minimize -sum
        radii = v[n*2:]
        return -np.sum(radii)

    # --- Initialization ---
    
    # Generate a hexagonal lattice pattern for initial centers
    # This is denser than a square grid
    centers_init = []
    radii_init = 0.1 # Initial guess radius
    
    # Grid parameters
    dx = 2 * radii_init
    dy = radii_init * np.sqrt(3)
    
    # Generate points
    # We want roughly 26 points.
    # A 5x5 grid has 25. Hexagonal can fit more.
    # Let's generate a larger grid and pick valid points, or construct rows.
    
    # Constructing rows manually for better control
    rows = [5, 5, 5, 5, 4, 2] # Sum = 26? 5*4 + 4 + 2 = 26. Yes.
    # Actually 5,5,5,5,5,1 is 26.
    # Let's try 5, 5, 5, 5, 5, 1.
    # Row heights: 0, dy, 2dy, 3dy, 4dy, 5dy.
    # Width available: 1. 
    # Row 1 (5 circles): x = 0.1, 0.3, 0.5, 0.7, 0.9 (if r=0.1)
    # Row 2 (offset): x = 0.2, 0.4, 0.6, 0.8, 1.0 (1.0 invalid). 
    # So strict hexagonal with r=0.1 might not fit 5 offset circles.
    # But we will optimize, so initial positions don't need to be perfect, just valid.
    
    # Let's create a valid initial configuration with smaller radius, say 0.08
    r_start = 0.08
    centers_list = []
    
    # Hexagonal packing layout
    # Row 0: 5 circles
    y = r_start
    for k in range(5):
        x = r_start + k * (2 * r_start)
        centers_list.append([x, y])
        
    # Row 1: 5 circles (offset)
    y += r_start * np.sqrt(3)
    offset = r_start
    for k in range(5):
        x = offset + k * (2 * r_start)
        # Check bounds
        if x - r_start >= 0 and x + r_start <= 1:
            centers_list.append([x, y])
            
    # Row 2: 5 circles
    y += r_start * np.sqrt(3)
    for k in range(5):
        x = r_start + k * (2 * r_start)
        if x - r_start >= 0 and x + r_start <= 1:
            centers_list.append([x, y])
            
    # Row 3: 5 circles (offset)
    y += r_start * np.sqrt(3)
    for k in range(5):
        x = offset + k * (2 * r_start)
        if x - r_start >= 0 and x + r_start <= 1:
            centers_list.append([x, y])
            
    # Row 4: 5 circles
    y += r_start * np.sqrt(3)
    for k in range(5):
        x = r_start + k * (2 * r_start)
        if x - r_start >= 0 and x + r_start <= 1:
            centers_list.append([x, y])
            
    # Row 5: We need 26 - 20 = 6 circles.
    # Try to fit 6 in a row?
    y += r_start * np.sqrt(3)
    # If we need 6, maybe reduce spacing or just place them.
    # For initialization, just place them randomly if not fitting in pattern, 
    # but optimization will fix it.
    # Let's just add remaining circles in a small grid or random valid spots.
    current_count = len(centers_list)
    while len(centers_list) < 26:
        # Try to place in Row 5
        y += r_start * np.sqrt(3) # Move down
        for k in range(6):
            x = r_start + k * (2 * r_start)
            if x - r_start >= 0 and x + r_start <= 1:
                centers_list.append([x, y])
            if len(centers_list) >= 26:
                break
        if len(centers_list) >= 26:
            break
            
    # If we still don't have 26 (due to bounds), scatter remaining
    if len(centers_list) < 26:
        # Random placement in valid region
        np.random.seed(42)
        for _ in range(26 - len(centers_list)):
            while True:
                x = np.random.uniform(r_start, 1 - r_start)
                y = np.random.uniform(r_start, 1 - r_start)
                # Check simple distance to existing (optional for init)
                centers_list.append([x, y])
                break

    centers_init = np.array(centers_list[:26])
    
    # --- Stage 1: Optimize Equal Radii ---
    # Variables: x1, y1, ..., x26, y26, r
    # Size: 52 + 1 = 53
    
    x0_equal = np.zeros(53)
    x0_equal[:52] = centers_init.flatten()
    x0_equal[52] = r_start
    
    cons_equal = get_constraints_equal(centers_init, r_start)
    
    # Use a robust solver. 'SLSQP' is good for constraints.
    # Bounds for x, y in [0, 1], r in [0, 0.5]
    bounds_equal = [(0, 1)] * 52 + [(0, 0.5)]
    
    res_equal = minimize(
        objective_equal, 
        x0_equal, 
        method='SLSQP', 
        bounds=bounds_equal, 
        constraints=cons_equal,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # Extract best equal radius config
    centers_eq = res_equal.x[:52].reshape(26, 2)
    r_eq = res_equal.x[52]
    
    # --- Stage 2: Optimize Unequal Radii ---
    # Variables: x1, y1, ..., x26, y26, r1, r2, ..., r26
    # Size: 52 + 26 = 78
    
    x0_unequal = np.zeros(78)
    x0_unequal[:52] = centers_eq.flatten()
    x0_unequal[52:] = r_eq * np.ones(26) # Start with equal radii
    
    cons_unequal = get_constraints_unequal(centers_eq, np.full(26, r_eq))
    
    bounds_unequal = [(0, 1)] * 52 + [(0, 0.5)] * 26
    
    res_unequal = minimize(
        objective_unequal, 
        x0_unequal, 
        method='SLSQP', 
        bounds=bounds_unequal, 
        constraints=cons_unequal,
        options={'maxiter': 2000, 'ftol': 1e-10}
    )
    
    # Final Result
    final_centers = res_unequal.x[:52].reshape(26, 2)
    final_radii = res_unequal.x[52:]
    final_sum = np.sum(final_radii)
    
    # Ensure non-negative radii (clipping small negative values due to numerical noise)
    final_radii = np.maximum(final_radii, 0.0)
    
    # Re-validate and fix if necessary
    # Sometimes optimization might slightly violate constraints due to tolerance.
    # We can do a final scaling if needed, but SLSQP usually respects bounds.
    
    # Check for any NaNs or invalids
    if np.isnan(final_centers).any() or np.isnan(final_radii).any():
        # Fallback to grid if optimization failed completely
        # Simple grid
        step = 1.0 / 6.0
        centers_fallback = []
        radii_fallback = []
        for i in range(6):
            for j in range(6):
                if i*6 + j < 26:
                    centers_fallback.append([step * (i + 0.5), step * (j + 0.5)])
                    radii_fallback.append(step * 0.4) # Safe radius
        final_centers = np.array(centers_fallback[:26])
        final_radii = np.array(radii_fallback[:26])
        final_sum = np.sum(final_radii)

    return final_centers, final_radii, final_sum

# The validation function is provided in the prompt, not defined here.
# But the run_packing function should be self-contained regarding logic.
