# sol_000312 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a46c309d) state=3cf4eb32 sum of radii=2.596666 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initialize centers in a 5x5 grid + 1 extra
    # This is a robust starting point better than purely random
    centers = []
    
    # Create a 5x5 grid
    grid_step = 0.9 / 4.5  # Scale to fit with some margin, actual fit will be optimized
    # Actually let's just use a uniform grid in [0.1, 0.9]
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            centers.append([x, y])
    
    # Add 26th circle
    # Place it in a gap or center? 
    # A 5x5 grid is quite dense. Let's place it at a random perturbed position or center.
    # Actually, for 26, a 6-row hexagonal-ish arrangement might be better, 
    # but let's just add one and let the optimizer find the spot.
    # Let's place it at (0.5, 0.5) but it's occupied. 
    # Let's place it at (0.5, 0.1) - bottom middle.
    centers.append([0.5, 0.1])
    
    centers = np.array(centers)
    n = 26
    
    # 2. Iterative Optimization
    # We will iterate: Solve LP for radii -> Compute forces -> Move centers
    num_iterations = 200
    step_size = 0.005
    
    for iteration in range(num_iterations):
        # --- Step A: Solve LP for Radii ---
        # Variables: r_0 ... r_25
        # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
        c_obj = -np.ones(n)
        
        # Constraints:
        # 1. r_i + r_j <= dist(i, j)  => r_i + r_j <= d_ij
        # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # 3. r_i >= 0
        
        # Prepare inequality constraints A_ub @ r <= b_ub
        # We will build these lists
        A_ub = []
        b_ub = []
        
        # Pairwise constraints
        # There are n*(n-1)/2 pairs. For n=26, ~325 constraints.
        # This is manageable.
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # r_i + r_j <= dist
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
        
        # Boundary constraints
        # For each circle i:
        # r_i <= x_i
        # r_i <= 1 - x_i
        # r_i <= y_i
        # r_i <= 1 - y_i
        # These are just upper bounds on individual variables, 
        # but we can add them to A_ub or use bounds. 
        # However, since r_i depends on centers which change, 
        # treating them as explicit constraints is safer if we want to track activity,
        # but for LP solver, bounds are easier.
        # Let's use bounds for variable r_i.
        # But wait, LP solver bounds are for variables.
        # r_i <= x_i is a constraint on r_i.
        # We can set bounds on r_i to [0, min(x_i, 1-x_i, y_i, 1-y_i)]
        
        bounds = []
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, 1-x, y, 1-y)
            # Ensure non-negative
            if max_r < 0: max_r = 0
            bounds.append((0, max_r))
            
        # Solve LP
        # Method 'highs' is robust
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
        
        if not res.success:
            # If LP fails (rare with valid init), break or continue
            # This might happen if constraints are contradictory (overlap > 0 dist?)
            # dist is always >= 0. If dist < 0 impossible.
            # If centers are same, dist=0, r_i+r_j <= 0 => r_i=r_j=0. Feasible.
            pass
        
        radii = res.x
        
        # --- Step B: Compute Forces and Move Centers ---
        # We want to move centers to relax the tightest constraints.
        # If r_i + r_j == dist(i, j), the circles are touching.
        # To allow larger radii, we should move them apart.
        # Force on i from j is proportional to vector (c_i - c_j) if touching.
        
        forces = np.zeros_like(centers)
        
        # Analyze slack to determine active constraints
        # Slack for pair (i, j) is dist - (r_i + r_j)
        # If slack is close to 0, apply force.
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                slack = dist - (radii[i] + radii[j])
                
                # If touching (within tolerance), apply repulsive force
                if slack < 1e-4: # Tolerance
                    # Direction from j to i
                    if dist > 1e-6:
                        dir_vec = (centers[i] - centers[j]) / dist
                    else:
                        dir_vec = np.random.rand(2) - 0.5 # Random push if coincident
                    
                    # Force magnitude? 
                    # If radii are large, force should be stronger?
                    # Or just constant repulsion.
                    force_mag = 1.0 
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag
        
        # Boundary forces
        # If r_i == x_i (touching left wall), push right.
        # r_i == 1 - x_i (touching right wall), push left.
        # etc.
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if np.abs(x - r) < 1e-4:
                forces[i][0] += 1.0
            # Right wall
            if np.abs((1 - x) - r) < 1e-4:
                forces[i][0] -= 1.0
            # Bottom wall
            if np.abs(y - r) < 1e-4:
                forces[i][1] += 1.0
            # Top wall
            if np.abs((1 - y) - r) < 1e-4:
                forces[i][1] -= 1.0
        
        # Apply forces
        # Normalize forces to avoid huge jumps?
        # Or just scale by step_size.
        # Check norm of forces
        norms = np.linalg.norm(forces, axis=1)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        unit_forces = forces / norms[:, np.newaxis]
        
        # Adaptive step size?
        # If iteration is high, reduce step size.
        current_step = step_size * (1.0 - iteration / num_iterations)
        if current_step < 0.0001: current_step = 0.0001
        
        centers += unit_forces * current_step
        
        # Clip centers to stay within [0, 1] strictly? 
        # Actually centers must be such that r >= 0, so x in [0, 1].
        # But if we push too hard, x might go outside [0,1]?
        # The boundary forces push inwards, so it should be stable.
        # Just clamp to be safe.
        centers = np.clip(centers, 1e-5, 1 - 1e-5)

    # 3. Final Radius Calculation
    # Re-solve LP one last time with final centers to get consistent radii
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1-x, y, 1-y)
        if max_r < 0: max_r = 0
        bounds.append((0, max_r))
        
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    
    if res.success:
        final_radii = res.x
    else:
        # Fallback: just set radii to small safe value if optimization failed
        final_radii = np.full(n, 0.01)

    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii
