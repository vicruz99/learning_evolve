# sol_000372 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b75b923f) state=6bb4cfe3 sum of radii=2.428956 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing():
    """
    Returns a packing of 26 circles in a unit square maximizing the sum of radii.
    """
    n = 26
    
    # Initial Configuration: 
    # A 5x5 grid (25 circles) plus one in the center, with small radii to ensure validity.
    # Grid points: 0.1, 0.3, 0.5, 0.7, 0.9
    # We scale down slightly to ensure they are well inside and separate initially.
    
    centers = []
    radii = []
    
    # Generate 5x5 grid
    coords = [0.2, 0.4, 0.6, 0.8] # Wait, let's use 0.2, 0.4, 0.6, 0.8? No, 5 points.
    # 5 points spaced evenly in [0.1, 0.9] -> 0.1, 0.3, 0.5, 0.7, 0.9
    grid_coords = np.linspace(0.1, 0.9, 5)
    
    count = 0
    for x in grid_coords:
        for y in grid_coords:
            centers.append([x, y])
            radii.append(0.05) # Initial radius
            count += 1
            if count >= 26:
                break
        if count >= 26:
            break
            
    # If we need more circles (we have 25), add one.
    # The loop above generates 25. We need 1 more.
    # Actually the loop runs 5*5 = 25 times.
    # Let's just append one in the center gap if needed, or just perturb.
    # The loop generates 25. Let's add the 26th.
    # But wait, the problem asks for 26.
    # My loop condition `if count >= 26` breaks after 26.
    # But 5x5 is 25. So it will finish the loop with 25 circles.
    # I need to ensure I have 26.
    
    # Let's restart initialization logic to be sure.
    centers = []
    radii = []
    
    # Use a hexagonal-like initialization or just random perturbation of a grid
    # Let's place 26 circles.
    # 5 rows. 6, 5, 6, 5, 4? No, width issues.
    # Let's just use a dense random initialization constrained to be valid.
    
    np.random.seed(42) # For reproducibility
    
    # Strategy: Place circles in a grid, then jitter.
    # We need 26 circles.
    # 5 rows of ~5 circles.
    # Let's place them.
    
    centers = []
    radii = []
    
    # Create a 5x5 grid (25 circles)
    x_grid = np.linspace(0.15, 0.85, 5) # Tighter packing
    y_grid = np.linspace(0.15, 0.85, 5)
    
    idx = 0
    for x in x_grid:
        for y in y_grid:
            centers.append([x, y])
            radii.append(0.04) # Small radius
            idx += 1
    
    # Add 26th circle in the center
    centers.append([0.5, 0.5])
    radii.append(0.04)
    
    centers = np.array(centers)
    radii = np.array(radii)
    
    # Now we optimize.
    # We will optimize variables: x1, y1, r1, x2, y2, r2, ...
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    # Constraints:
    # 1. r_i >= 0
    # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i (Inside square)
    # 3. dist(i,j) >= r_i + r_j
    
    # Flattened variables
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Constraints function for SLSQP
    # SLSQP expects constraints to be >= 0
    def objective(vars):
        # vars: x1, y1, ..., xn, yn, r1, ..., rn
        r = vars[2*n:]
        return -np.sum(r)

    # We need to handle constraints. SLSQP supports 'ineq' constraints.
    # However, with n=26, number of pairwise constraints is 26*25/2 = 325.
    # Plus 4*n = 104 boundary constraints.
    # Total ~430 constraints. This might be slow or fail to converge in one shot.
    
    # Alternative: Penalty Method inside objective?
    # Or just a simple iterative solver.
    
    # Let's try a simple iterative repulsion solver which is often more robust for packing.
    # We treat this as a physical system.
    
    # State: centers (N, 2), radii (N,)
    # We want to increase radii.
    # Force model:
    # Repulsion between circles: F ~ 1/d^2 or similar if overlapping.
    # Pressure to expand radii.
    
    # Let's implement a custom optimizer loop.
    
    # Current state
    C = centers.copy() # (N, 2)
    R = radii.copy()   # (N,)
    
    # Learning rate for radii expansion
    r_step = 0.001
    # Learning rate for position updates
    pos_step = 0.01
    
    # Number of iterations
    # We can run this for a fixed number of steps.
    iterations = 2000
    
    for step in range(iterations):
        # Adjust step sizes for annealing?
        current_r_step = r_step * (1.0 - step/iterations) 
        if current_r_step < 0.0001:
            current_r_step = 0.0001
            
        current_pos_step = pos_step * (1.0 - step/iterations)
        if current_pos_step < 0.00001:
            current_pos_step = 0.00001

        # Calculate forces/adjustments
        # We want to increase R.
        # If constraints are violated, we must decrease R or move C.
        
        # 1. Try to increase radii
        R += current_r_step
        
        # 2. Check constraints and apply corrective forces
        # We will accumulate displacement vectors for centers
        deltas = np.zeros_like(C)
        
        valid = True
        
        # Boundary constraints
        for i in range(n):
            x, y = C[i]
            r = R[i]
            
            # Check X
            if x - r < 0:
                # Push center right
                delta_x = - (x - r) + 0.0001 # Move right to satisfy
                deltas[i, 0] += delta_x
                valid = False
            elif x + r > 1:
                # Push center left
                delta_x = - (x + r - 1) - 0.0001
                deltas[i, 0] += delta_x
                valid = False
                
            # Check Y
            if y - r < 0:
                delta_y = - (y - r) + 0.0001
                deltas[i, 1] += delta_y
                valid = False
            elif y + r > 1:
                delta_y = - (y + r - 1) - 0.0001
                deltas[i, 1] += delta_y
                valid = False
        
        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = C[i] - C[j]
                dist = np.linalg.norm(dist_vec)
                sum_r = R[i] + R[j]
                
                if dist < sum_r:
                    # Overlap detected. Push apart.
                    # Distance needed: sum_r
                    # Current distance: dist
                    # Deficit: sum_r - dist
                    # Direction: normalize(dist_vec)
                    
                    if dist > 1e-9:
                        norm_vec = dist_vec / dist
                    else:
                        # Parallel circles? Random nudge
                        norm_vec = np.random.rand(2)
                        norm_vec /= np.linalg.norm(norm_vec)
                    
                    # Push apart by deficit/2 each? Or proportional to radius?
                    # Equal push is fine.
                    push = (sum_r - dist) / 2.0 + 0.0001 # Extra margin
                    
                    deltas[i] += norm_vec * push
                    deltas[j] -= norm_vec * push
                    valid = False
        
        # Apply displacements
        # Clamping is safer than direct addition to avoid oscillation?
        # Or just simple gradient step.
        C += deltas * 0.5 # Damping factor 0.5
        
        # Clamp centers to [0, 1] to prevent flying out
        C = np.clip(C, 0.0, 1.0)
        
        # If valid, we successfully expanded. If not, we moved to fix.
        # But we expanded R unconditionally. 
        # If overlap persists, R is too big.
        # We should perhaps scale R down if too many overlaps?
        # Or just let the position solver catch up.
        
        # To ensure convergence to a valid state with max R, 
        # we can check validity. If invalid, reduce R slightly?
        # But the loop above pushes them apart.
        
        # Let's add a mechanism: if too many overlaps, reduce R growth or shrink.
        overlap_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(C[i] - C[j])
                if dist < R[i] + R[j] - 1e-6:
                    overlap_count += 1
        
        if overlap_count > 10:
            # Too much overlap, reduce radii slightly to help escape
            R *= 0.99

    # Final cleanup: Ensure strict validity
    # The simulation might leave tiny overlaps.
    # We can run a final correction or just accept if validation passes.
    # But we need to return a valid packing.
    
    # Let's try to fix any remaining overlaps by reducing radii slightly
    # or moving centers.
    
    # Simple pass: reduce radii of overlapping pairs
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(C[i] - C[j])
                req = R[i] + R[j]
                if dist < req - 1e-9:
                    # Scale down radii to fit
                    scale = dist / req
                    R[i] *= scale
                    R[j] *= scale
                    changed = True
        
        # Also check boundaries
        for i in range(n):
            x, y = C[i]
            # Max radius allowed by boundary
            max_r_bound = min(x, 1-x, y, 1-y)
            if R[i] > max_r_bound + 1e-9:
                R[i] = max_r_bound
                changed = True

    # Final check of sum
    sum_radii = np.sum(R)
    
    return C, R, sum_radii
