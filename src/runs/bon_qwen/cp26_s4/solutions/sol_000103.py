# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 80fa60f2) state=adb8526d sum of radii=2.514342 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def generate_hexagonal_grid(n):
    """Generates an initial configuration of centers on a perturbed hexagonal grid."""
    centers = []
    rows = int(np.ceil(np.sqrt(n * 1.1547)))  # Estimate rows for hex packing
    # Hexagonal packing geometry
    y_step = 1.0 / (rows + 1)
    
    for r in range(rows):
        y = (r + 1) * y_step
        x_step = y_step * (2.0 / np.sqrt(3))
        offset = x_step / 2.0 if r % 2 == 1 else 0.0
        
        # Calculate valid x positions for this row
        x = offset
        while x <= 1.0 and len(centers) < n:
            if x >= 0:  # Ensure x is within bounds
                centers.append([x, y])
                if len(centers) >= n:
                    break
            x += x_step
    
    # If we didn't get enough circles, fall back to a dense grid fill
    while len(centers) < n:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        centers.append([x, y])
        
    return np.array(centers[:n])

def solve_radii_lp(centers):
    """Solves the LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Constraints: A_ub @ r <= b_ub
    constraints_A = []
    constraints_b = []
    
    # 1. Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # We add 4 constraints per circle, but can condense them if desired.
    # Keeping them separate is safer for the solver structure.
    for i in range(n):
        x, y = centers[i]
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(x)
        
        # r_i <= 1 - x_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(1.0 - x)
        
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(y)
        
        # r_i <= 1 - y_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(1.0 - y)

    # 2. Non-overlap constraints: r_i + r_j <= dist(c_i, c_j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints_A.append(row)
            constraints_b.append(dist)

    A_ub = np.array(constraints_A)
    b_ub = np.array(constraints_b)
    
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = -res.fun  # res.fun is min(-sum(r)), so -res.fun is max(sum(r))? 
                              # Wait, linprog minimizes c^T x. c = -1. min -sum(r) = -max sum(r).
                              # res.x contains the values of r.
            return res.x
        else:
            # Fallback if LP fails (e.g. infeasible due to numerical issues)
            return np.full(n, 1e-9)
    except Exception:
        return np.full(n, 1e-9)

def run_packing():
    # 1. Initialize Centers
    n = 26
    centers = generate_hexagonal_grid(n)
    
    # 2. Initial Radii Calculation
    radii = solve_radii_lp(centers)
    
    # 3. Optimization Loop
    # We iteratively move centers based on forces derived from tight constraints
    num_iterations = 500
    learning_rate = 0.01
    
    for it in range(num_iterations):
        # Recalculate radii for current centers to ensure validity and check tightness
        radii = solve_radii_lp(centers)
        
        forces = np.zeros_like(centers)
        
        # Analyze constraints to compute forces
        for i in range(n):
            r_i = radii[i]
            x_i, y_i = centers[i]
            
            # Boundary Forces
            # If circle is touching left wall (x_i approx r_i)
            if x_i - r_i < 1e-7:
                forces[i, 0] += 1.0
            # If touching right wall
            if (1.0 - x_i) - r_i < 1e-7:
                forces[i, 0] -= 1.0
            # If touching bottom wall
            if y_i - r_i < 1e-7:
                forces[i, 1] += 1.0
            # If touching top wall
            if (1.0 - y_i) - r_i < 1e-7:
                forces[i, 1] -= 1.0
                
            # Pairwise Forces
            for j in range(i + 1, n):
                r_j = radii[j]
                dist = np.linalg.norm(centers[i] - centers[j])
                # Check if touching or overlapping
                if dist < r_i + r_j + 1e-7:
                    # Push apart
                    vec = centers[i] - centers[j]
                    if dist > 1e-9:
                        force_vec = vec / dist
                        forces[i] += force_vec
                        forces[j] -= force_vec
                    else:
                        # Handle overlap of centers (random push)
                        rand_dir = np.random.randn(2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        forces[i] += rand_dir
                        forces[j] -= rand_dir

        # Update centers
        # Apply forces scaled by learning rate
        centers = centers + learning_rate * forces
        
        # Clip centers to square [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Decay learning rate slightly to settle
        learning_rate *= 0.995

    # Final Radii Calculation
    radii = solve_radii_lp(centers)
    
    # Ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        centers = np.random.rand(n, 2) * 0.8 + 0.1
        radii = np.full(n, 0.05)
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
