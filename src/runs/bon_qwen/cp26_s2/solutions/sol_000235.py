# sol_000235 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e58a758a) state=2d7569d1 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization, simulated annealing for optimization,
    and a linear program for final radius refinement.
    """
    n_circles = 26
    num_iterations = 5000
    initial_temp = 1.0
    final_temp = 1e-6
    cooling_rate = 0.9995
    
    # 1. Initialize Centers: Hexagonal Pattern
    # We aim for roughly 5 rows. 
    # To fit 26 circles, we can use row counts like 6, 5, 5, 5, 5 or 5, 6, 5, 5, 5.
    # A 6-5-5-5-5 distribution fits well in a hexagonal layout.
    rows_counts = [6, 5, 5, 5, 5]
    
    # Estimate initial radius to fit 6 circles in width 1 with some margin
    # Width for k circles is approx 2r + (k-1)*2r*cos(0) ? 
    # In hexagonal, row width is roughly determined by 2r + (k-1)*r*sqrt(3) if staggered? 
    # No, in a row, circles are distance 2r apart horizontally if aligned.
    # But in hexagonal packing, rows are shifted. 
    # Let's just place them on a grid and let optimization fix it.
    
    centers = np.zeros((n_circles, 2))
    idx = 0
    
    # Try to center the packing
    # Approximate spacing
    dy = 1.0 / 5.0 
    dx = 1.0 / 6.0 
    
    current_y = dy
    for r_idx, count in enumerate(rows_counts):
        # Shift even rows (0-indexed) by dx/2 to create hexagonal pattern
        offset = (dx / 2.0) if (r_idx % 2 == 1) else 0.0
        # Adjust y slightly to center the block
        y_pos = 0.5 + (r_idx - 2.0) * dy 
        
        # Calculate x positions to center the row
        # Total width occupied by 'count' circles with spacing dx
        # We want them to span roughly [dx/2, 1-dx/2]
        start_x = (1.0 - (count - 1) * dx) / 2.0
        
        for c in range(count):
            x_pos = start_x + c * dx + offset
            # Clamp to valid range initially
            centers[idx, 0] = np.clip(x_pos, 0.05, 0.95)
            centers[idx, 1] = np.clip(y_pos, 0.05, 0.95)
            idx += 1
            
    # Initial radii estimate
    radii = np.full(n_circles, 0.08)
    
    # 2. Simulated Annealing / Force Directed Optimization
    # We optimize centers and radii simultaneously
    # Objective: Maximize sum(radii)
    # Constraints: Non-overlap, inside square
    
    # Helper to calculate constraints violation and forces
    def get_forces_and_violations(centers, radii):
        forces = np.zeros_like(centers)
        violations = 0.0
        
        n = len(centers)
        for i in range(n):
            # Boundary forces
            x, y = centers[i]
            r = radii[i]
            
            # Push away from boundaries
            margin_x = x - r
            margin_y = y - r
            margin_x_right = 1.0 - (x + r)
            margin_y_top = 1.0 - (y + r)
            
            if margin_x < 0:
                forces[i, 0] += margin_x * 1000.0 # Strong repulsion
                violations += margin_x ** 2
            if margin_y < 0:
                forces[i, 1] += margin_y * 1000.0
                violations += margin_y ** 2
            if margin_x_right < 0:
                forces[i, 0] -= margin_x_right * 1000.0
                violations += margin_x_right ** 2
            if margin_y_top < 0:
                forces[i, 1] -= margin_y_top * 1000.0
                violations += margin_y_top ** 2
                
            # Interaction forces
            for j in range(i + 1, n):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Repulsive force proportional to overlap
                    force_mag = overlap * 500.0 
                    force_dir = dist_vec / dist
                    forces[i] += force_dir * force_mag
                    forces[j] -= force_dir * force_mag
                    violations += overlap ** 2
                elif dist < 1e-9:
                     # Prevent division by zero, push apart randomly
                     forces[i] += np.random.randn(2) * 10.0
                     forces[j] -= np.random.randn(2) * 10.0
                     violations += 1.0
                     
        return forces, violations

    # Optimization Loop
    current_sum_radii = np.sum(radii)
    best_sum_radii = -1.0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    temp = initial_temp
    
    # We will also try to grow radii
    growth_rate = 1.0001 
    
    for step in range(num_iterations):
        temp = initial_temp * (final_temp / initial_temp) ** (step / num_iterations)
        
        # Perturb centers
        move_magnitude = 0.02 * np.sqrt(temp)
        new_centers = centers + np.random.randn(n_circles, 2) * move_magnitude
        
        # Perturb radii (try to grow)
        # We can't just grow arbitrarily, but we can try to increase them slightly
        # and see if it's valid, or use forces to guide them.
        # Here we keep radii relatively stable but allow small jitter
        radii_jitter = 0.005 * np.sqrt(temp)
        new_radii = np.clip(radii + np.random.randn(n_circles) * radii_jitter, 0.0, 0.5)
        
        # Apply forces to push circles apart and expand
        forces, violations = get_forces_and_violations(new_centers, new_radii)
        
        # Update centers based on forces
        # If violations are high, forces dominate.
        # If violations are low, we might want to expand radii?
        # Simple update: new_centers += forces * scaling
        force_scale = 0.005 
        proposed_centers = new_centers + forces * force_scale
        
        # Propose a slight radius increase if valid
        # Or just keep new_radii
        
        # Accept/Reject Logic (Metropolis)
        # We want to maximize sum of radii, penalizing violations heavily
        # Objective = sum(radii) - penalty * violations
        
        penalty = 100.0
        current_obj = np.sum(radii) - penalty * violations
        proposed_obj = np.sum(new_radii) - penalty * violations # Re-eval violations for proposed? 
        # Actually, let's just use the forces to move and accept if better
        
        # Re-calculate forces for the move step? 
        # Let's just perform the move and check validity roughly
        # To be safe, we update centers and radii
        
        centers = proposed_centers
        # Try to expand radii slightly if there is room (heuristically)
        # If violations == 0, we can grow radii
        if violations < 1e-10:
             # Grow radii
             radii = radii * 1.0005 
        else:
             # Shrink radii if overlapping significantly to recover
             radii = radii * 0.99 
            
        # Clamp centers
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    centers = best_centers
    radii = best_radii

    # 3. Final Refinement with Linear Programming
    # Given fixed centers, maximize sum of radii
    # Variables: r_0, ..., r_25
    # Maximize sum(r_i)
    # Subject to:
    # r_i + r_j <= dist(i, j)
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # r_i >= 0
    
    n = 26
    c_obj = -np.ones(n) # Minimize -sum(r)
    
    # Inequality constraints A_ub @ r <= b_ub
    # 1. Distance constraints: r_i + r_j <= d_ij
    # 2. Boundary constraints: r_i <= bounds
    
    A_ub = []
    b_ub = []
    
    # Distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            dist = np.linalg.norm(centers[i] - centers[j])
            b_ub.append(dist)
            
    # Boundary constraints
    bounds_list = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        # Add constraint r_i <= max_r
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(max_r)
        
        # Bounds for LP solver (r_i >= 0)
        bounds_list.append((0, None))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list, method='highs')
    
    if res.success:
        final_radii = -res.x # Max sum, res.x is min -sum
        final_sum = np.sum(final_radii)
    else:
        final_radii = radii
        final_sum = np.sum(radii)
        
    # Verify and clamp any numerical issues
    # Ensure circles are inside
    for i in range(n):
        x, y = centers[i]
        r = final_radii[i]
        if r < 0: r = 0
        if x - r < 0: r = x
        if x + r > 1: r = 1 - x
        if y - r < 0: r = y
        if y + r > 1: r = 1 - y
        final_radii[i] = r

    # Re-validate overlaps after clamping (might need slight reduction)
    # Simple iterative reduction if overlap detected
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                req_sum = dist # r_i + r_j <= dist
                curr_sum = final_radii[i] + final_radii[j]
                if curr_sum > req_sum + 1e-12:
                    # Reduce both equally
                    excess = (curr_sum - req_sum) / 2.0
                    final_radii[i] -= excess
                    final_radii[j] -= excess
                    if final_radii[i] < 0: final_radii[i] = 0
                    if final_radii[j] < 0: final_radii[j] = 0
                    changed = True

    final_sum = np.sum(final_radii)
    return centers, final_radii, final_sum
