# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cae61cda) state=28aefa4d sum of radii=2.552043 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def solve_radii(centers):
    """
    Given fixed centers, solves the Linear Program to maximize the sum of radii
    subject to non-overlap and boundary constraints.
    """
    n = len(centers)
    # Variables: r_0, ..., r_{n-1}
    # Objective: max sum(r) -> min -sum(r)
    c = np.ones(n) * -1.0

    # Constraints: A_ub * r <= b_ub
    # We need to construct the matrix for:
    # 1. Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # 2. Pairwise constraints: r_i + r_j <= dist(i, j)
    
    n_pairs = n * (n - 1) // 2
    n_constraints = 4 * n + n_pairs
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    for i in range(n):
        x, y = centers[i]
        
        # r_i <= x
        A_ub[row, i] = 1.0
        b_ub[row] = x
        row += 1
        # r_i <= 1 - x
        A_ub[row, i] = 1.0
        b_ub[row] = 1.0 - x
        row += 1
        # r_i <= y
        A_ub[row, i] = 1.0
        b_ub[row] = y
        row += 1
        # r_i <= 1 - y
        A_ub[row, i] = 1.0
        b_ub[row] = 1.0 - y
        row += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = dist
            row += 1
            
    # Bounds for r: [0, infinity)
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
    else:
        radii = np.zeros(n)
        
    return radii, -res.fun

def generate_hex_lattice(n_circles):
    """
    Generates a hexagonal lattice of points within [0, 1] x [0, 1].
    """
    # Approximate spacing based on n
    # Area ~ 1. n * pi * r^2 * density ~ 1.
    # We just want a good spread.
    
    # Try to fit points. 
    # Hexagonal packing density is high.
    # Spacing s. Points at (i*s, j*s*sqrt(3)/2)
    
    points = []
    # Heuristic grid search for hex parameters
    best_n = 0
    best_spacing = 0.1
    
    # We can just generate a grid and pick first n
    # Hex grid:
    # row y = k * h
    # col x = m * w + (k % 2) * w/2
    
    h = 0.12
    w = 0.2
    pts = []
    
    y = 0.1 # margin
    while y < 0.9:
        x = 0.1
        row_offset = (int(y / h) % 2) * (w / 2)
        while x < 0.9:
            pts.append([x + row_offset, y])
            x += w
        y += h
        
    if len(pts) < n_circles:
        # Increase density
        pts = []
        h = 0.08
        w = 0.14
        y = 0.05
        while y < 0.95:
            x = 0.05
            row_offset = (int(y / h) % 2) * (w / 2)
            while x < 0.95:
                pts.append([x + row_offset, y])
                x += w
            y += h
            
    # Take first n_circles
    # Shuffle to break symmetry slightly
    pts = pts[:n_circles]
    return np.array(pts)

def run_packing():
    n = 26
    np.random.seed(123) # For reproducibility
    
    # 1. Initialization
    centers = generate_hex_lattice(n)
    
    # Add small random jitter to avoid symmetry issues
    centers += np.random.uniform(-0.01, 0.01, (n, 2))
    centers = np.clip(centers, 0.01, 0.99)
    
    best_centers = centers.copy()
    best_sum = 0.0
    
    # 2. Optimization Loop
    for iteration in range(2000):
        # Solve LP for current centers
        radii, current_sum = solve_radii(centers)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            
        # Calculate forces for next iteration
        forces = np.zeros_like(centers)
        
        # Force parameters
        repulsion_strength = 0.05
        wall_strength = 0.05
        
        # Pairwise forces (Repulsion if touching)
        for i in range(n):
            for j in range(i + 1, n):
                r_sum = radii[i] + radii[j]
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                
                # If distance is approximately equal to sum of radii, they are touching/constrained
                # We apply a force to push them apart to allow growth
                if dist < r_sum + 1e-4:
                    vec = centers[i] - centers[j]
                    if dist > 1e-8:
                        norm_vec = vec / dist
                        forces[i] += repulsion_strength * norm_vec
                        forces[j] -= repulsion_strength * norm_vec
        
        # Wall forces (Push away if constrained)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 1e-4:
                forces[i, 0] += wall_strength
            # Right wall
            if x + r > 1.0 - 1e-4:
                forces[i, 0] -= wall_strength
            # Bottom wall
            if y - r < 1e-4:
                forces[i, 1] += wall_strength
            # Top wall
            if y + r > 1.0 - 1e-4:
                forces[i, 1] -= wall_strength
                
        # Apply forces with decay
        step_size = 0.5 / (1 + iteration * 0.05)
        centers += step_size * forces
        
        # Boundary check for centers
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
        # Random perturbation (Simulated Annealing style) to escape local optima
        if iteration % 50 == 0:
            idx = np.random.randint(n)
            centers[idx] += np.random.uniform(-0.05, 0.05, 2)
            centers[idx] = np.clip(centers[idx], 1e-5, 1.0 - 1e-5)

    # Final calculation
    radii, final_sum = solve_radii(best_centers)
    return best_centers, radii, final_sum
