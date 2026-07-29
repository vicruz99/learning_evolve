# sol_000279 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 20c39dac) state=669a2ff4 sum of radii=1.038166 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Strategy:
    1. Initialize centers in a hexagonal grid pattern.
    2. Use a repulsion-based optimization to spread centers apart (maximizing min-distance).
    3. Solve a Linear Program to find optimal radii for the fixed centers.
    4. Iterate: Adjust centers based on LP sensitivity (or simple repulsion) to improve sum of radii.
    
    Returns:
        centers: (26, 2) array of (x, y) coordinates
        radii: (26,) array of radii
        sum_radii: float sum of all radii
    """
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We try to fit n points in a hexagonal lattice.
    # Estimate spacing based on sqrt(n)
    # 26 is close to 5*5=25. Let's try a 5x5 grid perturbed to hexagonal.
    
    # Start with a grid
    centers = np.zeros((n, 2))
    # Fill grid roughly 5x6 or similar
    cols = 6
    rows = 5
    x_step = 1.0 / (cols + 1)
    y_step = 1.0 / (rows + 1)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                # Hexagonal offset for odd rows
                x_off = x_step * (c + 1)
                y = y_step * (r + 1)
                if r % 2 == 1:
                    x_off += x_step / 2.0
                centers[idx] = [x_off, y]
                idx += 1
            else:
                break
        if idx >= n:
            break
    
    # 2. Repulsion Optimization (Force Directed)
    # We want to maximize the minimum distance between points and boundaries.
    # We use a simple iterative relaxation.
    
    dt = 0.01
    temperature = 0.1
    
    for step in range(500):
        forces = np.zeros_like(centers)
        
        # Repulsion between points
        # Use Coulomb-like repulsion but capped to avoid singularity
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff**2)
                dist = math.sqrt(dist_sq)
                if dist < 1e-6:
                    dist = 1e-6
                    diff = np.random.randn(2) * 1e-4 # jitter
                
                # Repulsive force inversely proportional to distance squared
                # We want to push them apart to increase minimum distance
                # A strong repulsion for close pairs helps maximize min-distance
                f_mag = 1.0 / (dist**2 + 0.01) # Softening
                f_vec = f_mag * diff
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Boundary repulsion
        # Push points away from walls to increase distance to boundary
        for i in range(n):
            x, y = centers[i]
            # Wall distances
            d_left = x
            d_right = 1.0 - x
            d_bottom = y
            d_top = 1.0 - y
            
            # Force magnitude increases as we get closer to wall
            f_x = 0.0
            f_y = 0.0
            
            if d_left < 0.2:
                f_x += 1.0 / (d_left**2 + 0.01)
            if d_right < 0.2:
                f_x -= 1.0 / (d_right**2 + 0.01)
            
            if d_bottom < 0.2:
                f_y += 1.0 / (d_bottom**2 + 0.01)
            if d_top < 0.2:
                f_y -= 1.0 / (d_top**2 + 0.01)
            
            forces[i] += [f_x, f_y]

        # Update positions
        # Normalize forces to prevent explosion, use small step
        max_f = np.max(np.linalg.norm(forces, axis=1))
        if max_f > 0:
            forces = forces / max_f
        
        centers += dt * forces
        
        # Clamp to [0, 1]
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
        # Reduce step size (cooling)
        dt *= 0.995

    # 3. Solve LP for radii given fixed centers
    # Maximize sum(r_i)
    # Subject to:
    # r_i + r_j <= dist(c_i, c_j)
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # r_i >= 0
    
    # LP Formulation:
    # Minimize -sum(r_i)
    # Variables: r_0, ..., r_25
    
    # Constraints matrix A_ub * x <= b_ub
    # Variables x = [r_0, ..., r_25]
    
    n_vars = n
    n_constraints = 0
    
    # Count constraints
    # Pairwise: n*(n-1)/2
    # Boundary: 4*n
    n_pairwise = n * (n - 1) // 2
    n_boundary = 4 * n
    total_constraints = n_pairwise + n_boundary
    
    A_ub = np.zeros((total_constraints, n_vars))
    b_ub = np.zeros(total_constraints)
    
    row_idx = 0
    
    # Pairwise constraints: r_i + r_j <= dist
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dist
            row_idx += 1
            
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = x
        row_idx += 1
        
        # r_i <= 1 - x
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = 1.0 - x
        row_idx += 1
        
        # r_i <= y
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = y
        row_idx += 1
        
        # r_i <= 1 - y
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = 1.0 - y
        row_idx += 1
        
    # Bounds for r_i >= 0
    bounds = [(0, None) for _ in range(n_vars)]
    
    # Objective: Minimize -sum(r_i) -> c = [-1, -1, ...]
    c_obj = np.ones(n_vars) * -1.0
    
    # Solve LP
    # Use high precision if possible, but default is usually fine
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
    else:
        # Fallback to simple calculation if LP fails (should not happen)
        radii = np.zeros(n)
        for i in range(n):
            r = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    r = min(r, np.linalg.norm(centers[i]-centers[j])) # Simplistic, assumes r_j=0? No.
            # This fallback is weak, but LP should work.
            # A better fallback for fixed centers is iterative shrinking.
            # Let's do iterative shrinking for fallback
            radii = np.full(n, 0.05) # Initial guess
            for _ in range(100):
                changed = False
                for i in range(n):
                    r_i = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
                    for j in range(n):
                        if i != j:
                            r_i = min(r_i, np.linalg.norm(centers[i]-centers[j]) - radii[j])
                    if r_i < radii[i] - 1e-9:
                        radii[i] = max(0, r_i)
                        changed = True
                if not changed:
                    break

    sum_radii = np.sum(radii)
    
    # Validation check
    # Although we trust the logic, let's ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Fallback to a safe grid packing if optimization failed completely
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        # 5x5 grid
        r_safe = 0.09
        c_idx = 0
        for r in range(5):
            for c in range(5):
                if c_idx < n:
                    centers[c_idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                    radii[c_idx] = r_safe
                    c_idx += 1
        # Last one?
        if c_idx < n:
             centers[c_idx] = [0.5, 0.5]
             radii[c_idx] = 0.01 # tiny
             c_idx += 1
        
        sum_radii = np.sum(radii)

    return centers, radii, sum_radii
