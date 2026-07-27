# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b137705a) state=f9355564 sum of radii=0.003443 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # 1. Initialization: 5x5 grid + 1 gap circle
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 5x5 grid (first 25 circles)
    count = 0
    for i in range(5):
        for j in range(5):
            centers[count, 0] = 0.1 + 0.2 * i
            centers[count, 1] = 0.1 + 0.2 * j
            radii[count] = 0.1
            count += 1
            
    # 26th circle in a gap (e.g., center of the first grid cell)
    # Grid centers are at 0.1, 0.3. Midpoint is 0.2.
    centers[25, 0] = 0.2
    centers[25, 1] = 0.2
    radii[25] = 0.0414  # Initial valid radius for the gap
    
    # 2. Force-Directed Optimization
    # Parameters
    dt = 0.005
    repulsion_strength = 5.0
    friction = 0.95
    n_steps = 2000
    
    # Compute distances matrix once to reuse logic (though distances change)
    # We will compute forces dynamically
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Calculate repulsion forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                sum_radii = radii[i] + radii[j]
                
                # If they overlap or are close, push apart
                # Use a force that is strong when tight, zero when far
                if dist > 0:
                    # "Tightness" factor
                    tightness = sum_radii / dist
                    if tightness > 1.0 - 1e-6: # Overlapping or touching
                        # Direction vector normalized
                        fx = dx / dist
                        fy = dy / dist
                        
                        # Force magnitude based on how much they need to separate
                        # We want to increase sum_radii, so we push them apart to create room
                        # Simple repulsion
                        force_mag = repulsion_strength * (tightness - 0.95) 
                        if force_mag < 0: force_mag = 0
                        
                        forces[i, 0] += fx * force_mag
                        forces[i, 1] += fy * force_mag
                        forces[j, 0] -= fx * force_mag
                        forces[j, 1] -= fy * force_mag

        # Apply forces with boundary clipping
        # Instead of moving centers and then clipping, we apply boundary forces
        for i in range(n):
            # Boundary repulsion (virtual walls)
            # Push away from boundaries if close
            x, y = centers[i]
            r = radii[i]
            
            # Left wall (x=0)
            if x - r < 0.001:
                forces[i, 0] += 10.0 * (0.001 - (x - r))
            # Right wall (x=1)
            if x + r > 0.999:
                forces[i, 0] -= 10.0 * ((x + r) - 0.999)
            # Bottom wall (y=0)
            if y - r < 0.001:
                forces[i, 1] += 10.0 * (0.001 - (y - r))
            # Top wall (y=1)
            if y + r > 0.999:
                forces[i, 1] -= 10.0 * ((y + r) - 0.999)
                
        # Update centers
        centers += forces * dt
        centers *= friction # Damping to help convergence
        
        # Hard clip centers to [0, 1]
        np.clip(centers, 0, 1, out=centers)
        
        # Update radii to be maximal allowed by current positions (Greedy local expansion)
        # This step tries to inflate circles as the gaps open up
        for i in range(n):
            max_r = 1.0
            # Boundary constraints
            max_r = min(max_r, centers[i, 0])
            max_r = min(max_r, 1.0 - centers[i, 0])
            max_r = min(max_r, centers[i, 1])
            max_r = min(max_r, 1.0 - centers[i, 1])
            
            # Neighbor constraints
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                max_r = min(max_r, dist - radii[j])
            
            # Clamp radius to be at least previous radius (prevent shrinking)
            # But also respect the calculated max
            # Actually, in force-directed, radii might need to adjust dynamically.
            # We set it to the calculated max to maximize sum.
            if max_r < 0: max_r = 0 # Should not happen if valid
            radii[i] = max(radii[i], max_r) # Grow if possible, keep if not
            
            # Note: Growing one circle might violate constraints for others.
            # The force repulsion handles this by pushing centers apart.
            # However, for stability, we might just clamp to max allowed.
            radii[i] = max_r # Aggressively fill space.

    # 3. Final LP Optimization to maximize sum of radii given final centers
    # This ensures we extract the absolute maximum sum for the converged positions
    c = -np.ones(n) # Maximize sum(r) -> Minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    # Constraints: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    # Constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    # These are bounds in LP, but we can handle them as A_ub or bounds
    # Let's use bounds for r_i >= 0 and constraints for upper bounds if needed,
    # but standard LP bounds are easier.
    
    # However, standard linprog bounds are (lower, upper) for each variable.
    # But r_i depends on x_i which is fixed now.
    # So for each i, r_i <= min(x_i, 1-x_i, y_i, 1-y_i).
    
    bounds = []
    for i in range(n):
        max_r_boundary = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0, max_r_boundary))
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        optimal_radii = res.x
    else:
        # Fallback to previous radii if LP fails
        optimal_radii = radii
        
    # Ensure radii are valid (non-negative)
    optimal_radii = np.maximum(optimal_radii, 0)
    
    sum_radii = np.sum(optimal_radii)
    
    return centers, optimal_radii, sum_radii
