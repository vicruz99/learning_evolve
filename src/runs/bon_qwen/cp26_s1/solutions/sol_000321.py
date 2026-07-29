# sol_000321 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 78c934c9) state=9012c6ac sum of radii=2.501887 correctness=1.0
# stdout(first 200): Iter 0, Sum Radii: 2.1716 Iter 20, Sum Radii: 2.4340 Iter 40, Sum Radii: 2.4813 Iter 60, Sum Radii: 2.4941 Iter 80, Sum Radii: 2.5001 Iter 100, Sum Radii: 2.5017 Iter 120, Sum Radii: 2.5018 Iter 140, 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import linprog

def compute_pairwise_distances(centers):
    """Compute squared Euclidean distances between all pairs of centers."""
    n = centers.shape[0]
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dists[i, j] = d
            dists[j, i] = d
    return dists

def solve_radii_lp(centers, radii_lower_bound=1e-9):
    """
    Solve the Linear Programming problem to maximize sum of radii
    given fixed centers.
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c_obj = np.ones(n)
    
    # Constraints:
    # 1. r_i + r_j <= dist(i, j) for all i < j
    # 2. r_i <= distance to boundaries for all i
    # 3. r_i >= 0 (handled by bounds)
    
    n_pairs = n * (n - 1) // 2
    n_bdry = 4 * n
    n_constraints = n_pairs + n_bdry
    
    A = np.zeros((n_constraints, n))
    b = np.zeros(n_constraints)
    
    idx = 0
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < 1e-12:
                dist = 1e-12 # Prevent 0 distance issues
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = dist
            idx += 1
            
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r <= x (left boundary)
        A[idx, i] = 1.0
        b[idx] = x
        idx += 1
        # r <= 1 - x (right boundary)
        A[idx, i] = 1.0
        b[idx] = 1.0 - x
        idx += 1
        # r <= y (bottom boundary)
        A[idx, i] = 1.0
        b[idx] = y
        idx += 1
        # r <= 1 - y (top boundary)
        A[idx, i] = 1.0
        b[idx] = 1.0 - y
        idx += 1
        
    bounds = [(0.0, None)] * n
    
    # Use 'highs' method if available, fallback to 'interior-point'
    try:
        res = linprog(-c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    except ValueError:
        res = linprog(-c_obj, A_ub=A, b_ub=b, bounds=bounds, method='interior-point')
        
    if res.success:
        return res.x
    else:
        # Fallback: small radii
        return np.full(n, radii_lower_bound)

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We need 26 points. 
    # A 6x5 grid has 30 points. We will select 26.
    # Or we can generate a hexagonal pattern directly.
    # Let's try a dense packing pattern.
    # Rows with 5 and 6 circles alternating?
    # 5, 6, 5, 6, 4 = 26.
    
    centers = []
    row_y = 0.0
    # We will determine spacing later, initially just use unit square distribution
    # Let's use a regular grid for robust start, then optimize.
    # 6 rows, 5 cols.
    rows = 6
    cols = 5
    x_spacing = 1.0 / cols
    y_spacing = 1.0 / rows
    
    # Generate points
    points = []
    for r in range(rows):
        for c in range(cols):
            # Offset every other row for hexagonal packing feel
            if r % 2 == 1:
                x = (c + 0.5) * x_spacing
            else:
                x = c * x_spacing
            y = r * y_spacing + y_spacing / 2 # Center in cell
            # Keep within bounds [0, 1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
    
    # We have 30 points. Take 26.
    # Remove corners to reduce boundary pressure?
    # Just take first 26.
    centers = np.array(points[:n])
    
    # 2. Optimization Loop
    # We will iteratively improve centers.
    
    # Learning rate for center updates
    alpha = 0.05 
    decay = 0.95
    
    for iteration in range(200):
        # Solve for optimal radii
        radii = solve_radii_lp(centers)
        current_sum = np.sum(radii)
        
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Threshold for tight constraints
        tol = 1e-6
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < 1e-12: continue
                
                sum_r = radii[i] + radii[j]
                gap = dist - sum_r
                
                # If constraint is tight (gap <= tol), apply repulsion
                # Force magnitude proportional to how tight it is? 
                # Or just constant repulsion to separate them.
                # If gap is negative (overlap in radii sum vs dist, though LP ensures sum_r <= dist),
                # it means they are touching.
                # Actually LP ensures sum_r <= dist. So gap >= 0.
                # If gap is small, they are touching.
                if gap < 1e-4: # Tight
                    # Direction i -> j is (c_j - c_i)
                    # We want to move i away from j: direction (c_i - c_j)
                    vec = centers[i] - centers[j]
                    # Normalize
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        dir_ij = vec / norm
                        # Force magnitude: push them apart
                        # Stronger push if they are larger?
                        mag = 1.0 
                        forces[i] += mag * dir_ij
                        forces[j] -= mag * dir_ij

        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left: r <= x. If tight, push right (+x)
            if x - r < 1e-4:
                forces[i, 0] += 1.0
            # Right: r <= 1-x => x <= 1-r. If tight, push left (-x)
            if (1.0 - x) - r < 1e-4:
                forces[i, 0] -= 1.0
            # Bottom: r <= y. If tight, push up (+y)
            if y - r < 1e-4:
                forces[i, 1] += 1.0
            # Top: r <= 1-y. If tight, push down (-y)
            if (1.0 - y) - r < 1e-4:
                forces[i, 1] -= 1.0
        
        # Apply forces
        # Scale force by alpha and maybe by radii size?
        # Larger circles might need more room?
        # Let's just apply to centers.
        
        # Normalize forces to avoid explosion?
        # Or just clip.
        max_force = 0.1
        forces = np.clip(forces, -max_force, max_force)
        
        centers += alpha * forces
        
        # Project back to valid region [0, 1] x [0, 1]
        # Actually centers should stay inside.
        centers[:, 0] = np.clip(centers[:, 0], 0.0, 1.0)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, 1.0)
        
        # Decay alpha
        alpha *= decay
        
        if iteration % 20 == 0:
            print(f"Iter {iteration}, Sum Radii: {current_sum:.4f}")

    # Final clean-up solve
    radii = solve_radii_lp(centers)
    
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Note: The function run_packing must be defined as requested.
# The code above defines helper functions and the main function.
# I will wrap it properly.
