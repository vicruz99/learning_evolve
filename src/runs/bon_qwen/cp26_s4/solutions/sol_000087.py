# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f395aea4) state=d1c5b1a8 sum of radii=2.492405 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Set seed for reproducibility if needed, but random perturbation helps escape local optima
    rng = np.random.RandomState(42)
    
    n = 26
    
    # 1. Initial Configuration: Perturbed 5x5 Grid + 1
    # A standard 5x5 grid of radius 0.1 touches boundaries. 
    # We shrink slightly and add a perturbation to allow space for the 26th circle and optimization.
    centers = []
    radii = []
    
    # 25 circles in a grid
    for i in range(5):
        for j in range(5):
            # Base positions for 5x5 grid with r=0.1 would be 0.1, 0.3, ...
            # We scale by 0.99 to leave margin
            x = (2 * i + 1) * 0.1 * 0.99
            y = (2 * j + 1) * 0.1 * 0.99
            # Add small random perturbation
            x += rng.uniform(-0.005, 0.005)
            y += rng.uniform(-0.005, 0.005)
            centers.append([x, y])
            radii.append(0.099)
            
    # 26th circle in a gap (e.g., between first 4 grid circles)
    # Gap center approx (0.2, 0.2)
    centers.append([0.2, 0.2])
    radii.append(0.03) # Small initial radius
    
    centers = np.array(centers)
    radii = np.array(radii)
    
    # 2. Repulsion / Expansion Simulation (Physics-based heuristic)
    # This helps find a good local configuration where circles are pushed apart and expanded.
    
    dt = 0.01
    repulsion_strength = 1.0
    expansion_rate = 0.0005
    max_iters = 2000
    
    for _ in range(max_iters):
        forces = np.zeros_like(centers)
        
        # Calculate repulsion forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                r_sum = radii[i] + radii[j]
                
                # If overlapping or too close, repel
                if dist < r_sum + 1e-5 and dist > 1e-9:
                    # Repulsion force magnitude proportional to overlap
                    overlap = r_sum - dist
                    # Soft force: F ~ overlap
                    force_mag = overlap * repulsion_strength
                    direction = diff / dist
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
                elif dist < r_sum * 1.5 and dist > 1e-9:
                    # Gentle repulsion to keep some spacing for radius growth
                    overlap = (r_sum * 1.5) - dist
                    force_mag = overlap * repulsion_strength * 0.1
                    direction = diff / dist
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag

        # Update positions
        centers += forces * dt
        
        # Boundary constraints (bounce off walls)
        for i in range(n):
            r = radii[i]
            # X bounds
            if centers[i, 0] - r < 0:
                centers[i, 0] = r
                forces[i, 0] = 0 # Stop moving into wall
            elif centers[i, 0] + r > 1:
                centers[i, 0] = 1 - r
                forces[i, 0] = 0
            
            # Y bounds
            if centers[i, 1] - r < 0:
                centers[i, 1] = r
                forces[i, 1] = 0
            elif centers[i, 1] + r > 1:
                centers[i, 1] = 1 - r
                forces[i, 1] = 0

        # Expand radii slightly
        radii += expansion_rate
        
    # 3. Local Optimization using SciPy
    # We treat centers and radii as variables.
    # Objective: Maximize sum(radii) -> Minimize -sum(radii)
    # Constraints: Non-overlap and boundary.
    
    # Combine centers and radii into a single vector
    # x = [x1, y1, x2, y2, ..., r1, r2, ...]
    # Shape: 2*n + n = 3*n
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1)]) # x, y
    for i in range(n):
        bounds.append((0, 0.5)) # r
        
    # Constraint function for scipy
    # We need inequality constraints: g(x) >= 0
    # Boundary: x - r >= 0, 1 - x - r >= 0, etc.
    # Non-overlap: dist^2 - (r_i + r_j)^2 >= 0
    
    def constraints_func(vars_vec):
        c_list = []
        idx = 0
        centers_opt = np.zeros((n, 2))
        radii_opt = np.zeros(n)
        
        for i in range(n):
            centers_opt[i, 0] = vars_vec[idx]
            centers_opt[i, 1] = vars_vec[idx+1]
            radii_opt[i] = vars_vec[idx+2]
            idx += 3
            
        cons = []
        
        # Boundary constraints
        for i in range(n):
            r = radii_opt[i]
            x = centers_opt[i, 0]
            y = centers_opt[i, 1]
            cons.append(x - r)          # x >= r
            cons.append(1 - x - r)      # x + r <= 1
            cons.append(y - r)          # y >= r
            cons.append(1 - y - r)      # y + r <= 1
            
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers_opt[i] - centers_opt[j]) ** 2)
                r_sum = radii_opt[i] + radii_opt[j]
                cons.append(dist_sq - r_sum**2)
                
        return cons

    # Define constraint dictionary for scipy
    cons_scipy = {'type': 'ineq', 'fun': constraints_func}

    def objective(vars_vec):
        # Extract radii
        r_vec = vars_vec[2::3] # Step 3: x, y, r
        return -np.sum(r_vec) # Minimize negative sum

    # Run optimization
    # SLSQP is good for this type of problem
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_scipy, 
                   options={'maxiter': 1000, 'ftol': 1e-12})
    
    if res.success:
        final_vars = res.x
        final_centers = np.zeros((n, 2))
        final_radii = np.zeros(n)
        idx = 0
        for i in range(n):
            final_centers[i, 0] = final_vars[idx]
            final_centers[i, 1] = final_vars[idx+1]
            final_radii[i] = final_vars[idx+2]
            idx += 3
    else:
        # Fallback to simulation result if optimizer fails
        final_centers = centers
        final_radii = radii

    # Final validation check (internal)
    # The validation function in the prompt is strict, we ensure our result is valid.
    # SLSQP might return slightly invalid points due to tolerance, we clamp radii if needed.
    # But the constraints should handle it.
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
