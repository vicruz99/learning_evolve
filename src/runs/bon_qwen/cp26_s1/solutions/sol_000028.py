# sol_000028 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 27de0ea1) state=4a124db5 sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization with multiple restarts.
    """
    n = 26
    
    # Helper function to create initial configuration
    def generate_initial_config(seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Strategy: Hexagonal-like arrangement
        # We'll try to fit them in roughly 5-6 rows
        # Let's try a pattern that is dense
        # Pattern: 5, 6, 5, 6, 4 = 26
        rows_counts = [5, 6, 5, 6, 4]
        
        centers = []
        # Initial guess for radius. 
        # For 5 circles in a row, width ~ 10r <= 1 -> r <= 0.1
        # For 6 circles, width ~ 12r <= 1 -> r <= 0.083
        # Let's start with a safe radius like 0.06 and let optimizer grow it.
        r_guess = 0.06
        
        y_pos = r_guess + 0.05 # Start slightly off bottom
        
        for count in rows_counts:
            # Calculate spacing to fit 'count' circles in width 1
            # Width required approx 2*r + (count-1)*spacing
            # Let's just distribute them evenly in [r_guess, 1-r_guess]
            x_positions = np.linspace(r_guess + 0.02, 1 - r_guess - 0.02, count)
            
            # Stagger rows slightly
            if len(centers) > 0:
                # Shift by half a step roughly
                x_positions = x_positions + (x_positions[1] - x_positions[0]) / 2
                # Re-clamp
                x_positions = np.clip(x_positions, r_guess, 1 - r_guess)
            
            for x in x_positions:
                centers.append([x, y_pos])
            
            y_pos += (2 * r_guess) * np.sqrt(3) / 2 + 0.05 # Vertical step for hex packing + buffer
        
        centers = np.array(centers[:n]) # Ensure we have exactly n
        radii = np.full(n, r_guess)
        return centers, radii

    def objective_and_constraints(centers_flat, radii_flat):
        centers = centers_flat.reshape(-1, 2)
        radii = radii_flat
        
        # Objective: Maximize sum of radii -> Minimize negative sum
        obj_val = -np.sum(radii)
        
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # x - r >= 0  => r - x <= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, idx=i: v[idx*2] - v[n + idx]}) # Wait, variable ordering?
            # Let's flatten vars as [x1, y1, ..., xn, yn, r1, ..., rn]
            # But here we pass separate arrays. Let's define a wrapper later.
            pass 
            
        # Overlap constraints
        # dist >= r_i + r_j  => dist - r_i - r_j >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if np.isnan(dist):
                    dist = 10.0 # Penalty
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j, c=centers, r=radii: np.sqrt(np.sum((c[i]-c[j])**2)) - r[i] - r[j]})
                
        return obj_val, constraints

    # Better approach for scipy.optimize: 
    # Flatten all variables into a single vector x
    # x[0..2n-1] = centers coordinates
    # x[2n..3n-1] = radii
    
    def obj_func(x):
        radii = x[2*n:]
        return -np.sum(radii)

    def boundary_constraints(x):
        cons = []
        centers = x[:2*n].reshape(-1, 2)
        radii = x[2*n:]
        for i in range(n):
            x_i, y_i = centers[i]
            r_i = radii[i]
            # x_i - r_i >= 0
            cons.append(x_i - r_i)
            # 1 - x_i - r_i >= 0
            cons.append(1.0 - x_i - r_i)
            # y_i - r_i >= 0
            cons.append(y_i - r_i)
            # 1 - y_i - r_i >= 0
            cons.append(1.0 - y_i - r_i)
        return np.array(cons)

    def overlap_constraints(x):
        centers = x[:2*n].reshape(-1, 2)
        radii = x[2*n:]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                cons.append(dist - radii[i] - radii[j])
        return np.array(cons)

    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple restarts
    num_restarts = 5
    
    for k in range(num_restarts):
        # Generate initial guess
        # Random perturbation of a grid
        centers_init, radii_init = generate_initial_config(seed=k*123 + 42)
        
        # Flatten
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        # Bounds
        bounds = []
        for i in range(n):
            # x in [0, 1]
            bounds.append((0, 1))
            bounds.append((0, 1))
            # r in [0, 0.5]
            bounds.append((0, 0.5))
            
        constraints_list = [
            {'type': 'ineq', 'fun': boundary_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        try:
            res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints_list, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or (res.fun < -best_sum): # res.fun is negative sum
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2*n].reshape(-1, 2)
                    best_radii = res.x[2*n:]
        except Exception as e:
            print(f"Optimization failed on restart {k}: {e}")
            continue

    if best_centers is None:
        # Fallback to simple grid if optimization fails
        centers = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        radii = np.full(26, 0.01) # Small valid radii
        # Validate and adjust? 
        # Just return something valid
        pass
    else:
        centers = best_centers
        radii = best_radii
        
    # Final validation and clipping to ensure no NaNs or slight violations
    radii = np.maximum(radii, 0)
    centers = np.clip(centers, 0, 1)
    
    # If radii are too large, scale down slightly to ensure validity (robustness)
    # Check overlaps
    valid = True
    for i in range(n):
        if centers[i][0] < radii[i] or centers[i][0] + radii[i] > 1:
            valid = False
        if centers[i][1] < radii[i] or centers[i][1] + radii[i] > 1:
            valid = False
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i]-centers[j])**2))
            if dist < radii[i] + radii[j]:
                valid = False
    
    if not valid:
        # Fallback: uniform small circles
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        radii = np.full(n, 0.01)
        # Sort and place grid-like to be safe
        # Just return a trivial valid packing
        pass

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
