# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eaaa636a) state=37b21adf sum of radii=2.513220 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)

    # 1. Initialization: Hexagonal packing grid
    # We create a grid and select the first 26 points, then center them.
    # Hexagonal packing has rows shifted by 0.5 * spacing.
    spacing = 0.22 # Initial guess slightly larger to avoid immediate overlap issues
    points = []
    rows = 8
    for r in range(rows):
        y = r * spacing * np.sqrt(3) / 2
        num_in_row = 1 if r % 2 == 1 else 2 # Just a simple alternating pattern to get points
        # Actually, let's just generate a dense grid and pick 26
        pass

    # Better initialization: Random points within the square, slightly repulsed
    centers = np.random.rand(n, 2)
    
    # Initial radii guess (small)
    radii = np.full(n, 0.05)

    # 2. Optimization Loop
    # We will use a simple hill-climbing / force-directed approach.
    # In each step, we fix centers, solve LP for max radii, 
    # then compute forces to move centers to increase sum of radii.
    
    num_iterations = 2000
    initial_step_size = 0.02
    
    # Precompute pair indices
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    for iter_idx in range(num_iterations):
        step_size = initial_step_size * (0.995 ** iter_idx)
        
        # Current centers state
        current_centers = centers.copy()
        
        # --- LP Step: Maximize sum of radii given centers ---
        # Variables: r_1, ..., r_26
        # Maximize sum(r)
        # Subject to:
        # 1. r_i >= 0
        # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # 3. r_i + r_j <= dist(c_i, c_j)
        
        # LP formulation: Minimize -sum(r)
        # f = -1 * ones(n)
        # A_ub * r <= b_ub
        
        c_obj = -np.ones(n)
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= x_i, etc.
        # r_i - x_i <= 0  =>  1*r_i <= x_i
        # -r_i <= -x_i (handled by lower bound? No, r_i >= 0 is lower bound. Upper bound is variable)
        # Wait, standard form is A_ub x <= b_ub.
        # We have upper bounds on r_i.
        
        # We can use bounds in linprog for r_i >= 0.
        # But we have individual upper bounds depending on coordinates.
        # r_i <= x_i  =>  [1, 0, ...] * r <= x_i
        # r_i <= 1-x_i => [1, 0, ...] * r <= 1-x_i
        # ...
        
        # Matrix A_ub will have size (4*n + n*(n-1)/2) x n
        
        # Boundary constraints
        for i in range(n):
            # r_i <= x_i
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(current_centers[i, 0])
            
            # r_i <= 1 - x_i
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0 - current_centers[i, 0])
            
            # r_i <= y_i
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(current_centers[i, 1])
            
            # r_i <= 1 - y_i
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0 - current_centers[i, 1])
            
        # Pairwise distance constraints: r_i + r_j <= dist_ij
        for (i, j) in pair_indices:
            dist = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None)] * n
        
        # Solve LP
        try:
            res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                current_radii = res.x
                current_sum = np.sum(current_radii)
            else:
                # Fallback if LP fails (shouldn't happen with r=0 feasible)
                current_radii = np.zeros(n)
                current_sum = 0.0
        except Exception:
            current_radii = np.zeros(n)
            current_sum = 0.0

        # Track best solution found so far
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()

        # --- Force Calculation Step ---
        # We want to move centers to increase the sum of radii.
        # Sensitivity analysis suggests that if a constraint is tight (r_i + r_j approx dist),
        # moving centers apart increases dist, potentially increasing r.
        
        forces = np.zeros_like(current_centers)
        
        # Check tightness of pairwise constraints
        # Slack = dist - (r_i + r_j)
        # If slack is small, we are constrained by this pair.
        # We should apply repulsive force.
        
        for idx, (i, j) in enumerate(pair_indices):
            r_sum = current_radii[i] + current_radii[j]
            dist = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
            
            # Avoid division by zero
            if dist < 1e-9:
                dist = 1e-9
                direction = np.random.rand(2) - 0.5
            else:
                direction = (current_centers[i] - current_centers[j]) / dist
            
            slack = dist - r_sum
            
            # If slack is very small (constraint active), push apart
            # Heuristic: force magnitude inversely related to slack, but capped
            if slack < 1e-4:
                # Tight constraint
                force_mag = 1.0 
                # Scale by radii to handle size differences?
                # Larger circles might need more space?
                forces[i] += force_mag * direction
                forces[j] -= force_mag * direction
            elif slack < 0.05: # Proximity force
                # Not strictly tight, but close. Gentle push.
                force_mag = slack * 10.0 
                forces[i] += force_mag * direction
                forces[j] -= force_mag * direction

        # Boundary constraints forces
        for i in range(n):
            r = current_radii[i]
            x, y = current_centers[i]
            
            # If r is close to x, push x positive (away from 0)
            if x - r < 1e-4:
                forces[i, 0] += 1.0
            
            # If r is close to 1-x, push x negative (away from 1)
            if (1 - x) - r < 1e-4:
                forces[i, 0] -= 1.0
                
            if y - r < 1e-4:
                forces[i, 1] += 1.0
                
            if (1 - y) - r < 1e-4:
                forces[i, 1] -= 1.0

        # Update centers
        # Normalize forces to prevent huge jumps
        norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_forces[norm_forces == 0] = 1.0
        # Don't normalize direction, just scale by step size
        # Actually, raw forces might be large. Let's scale by step_size.
        
        centers += forces * step_size
        
        # Clip to valid region [0, 1] (though centers can be anywhere, 
        # if center < 0, radius will be clipped to 0 by LP, which is bad for sum.
        # Keeping centers in [0,1] is safe, but technically center can be at -0.1 with r=0?
        # No, constraint r <= x means if x<0, r<=negative -> r=0.
        # So keeping centers in [0,1] is optimal for maximizing r.
        centers = np.clip(centers, 0.0, 1.0)

    # Final LP solve on best centers to ensure consistency
    current_centers = best_centers
    A_ub = []
    b_ub = []
    for i in range(n):
        row = np.zeros(n); row[i] = 1.0; A_ub.append(row); b_ub.append(current_centers[i, 0])
        row = np.zeros(n); row[i] = 1.0; A_ub.append(row); b_ub.append(1.0 - current_centers[i, 0])
        row = np.zeros(n); row[i] = 1.0; A_ub.append(row); b_ub.append(current_centers[i, 1])
        row = np.zeros(n); row[i] = 1.0; A_ub.append(row); b_ub.append(1.0 - current_centers[i, 1])
    for (i, j) in pair_indices:
        dist = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
        row = np.zeros(n); row[i] = 1.0; row[j] = 1.0; A_ub.append(row); b_ub.append(dist)
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    res = opt.linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    final_radii = res.x if res.success else best_radii
    final_sum = np.sum(final_radii)
    
    # Just in case the last step didn't update best, re-validate
    # But best_centers/radii track the max sum found during iterations.
    # However, we should return the radii corresponding to the centers.
    # The loop updated best_radii inside.
    
    return best_centers, best_radii, float(np.sum(best_radii))
