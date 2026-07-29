# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f550adc) state=766c1764 sum of radii=2.607808 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    centers = []
    # Parameters for hexagonal packing
    # We want to cover the square [0,1]x[0,1]
    # Spacing roughly 0.2
    
    # Generate points
    points = []
    # Vertical spacing for hex grid: height = radius * sqrt(3)
    # Let's assume initial radius approx 0.1. 
    # dy = 0.1 * sqrt(3) approx 0.1732
    # dx = 0.2
    # But we don't know r yet, so just use a dense grid and pick 26
    # Actually, let's just use a uniform grid first, then optimize?
    # Hexagonal is better.
    
    # Let's try to fit rows.
    # If we have 5 rows, dy approx 1/4 = 0.25
    # If we have 6 rows, dy approx 1/5 = 0.2
    # Let's try 6 rows.
    
    # To get 26 circles, maybe 5, 5, 5, 5, 5, 1? Or 5, 5, 6, 5, 5?
    # Let's generate a pattern and select 26.
    
    pts = []
    # Row 0 to 5
    for j in range(6):
        y = 0.1 + j * 0.16 # spacing slightly less than 0.2 to fit 6 rows
        # shift odd rows
        shift = 0.1 if j % 2 == 1 else 0.0
        for k in range(5):
            x = 0.1 + k * 0.18 + shift
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
    
    # We might have more or less than 26.
    # If less, add random or fill grid. If more, trim.
    # Let's ensure we have at least 26 good points.
    # If pts has < 26, let's add some.
    if len(pts) < 26:
        # Fill with a simple grid
        grid_step = 0.12
        for gy in np.arange(0.1, 1.0, grid_step):
            for gx in np.arange(0.1, 1.0, grid_step):
                pts.append([gx, gy])
                if len(pts) >= 30: break
            if len(pts) >= 30: break
            
    # Select 26 points. 
    # Prefer points that are somewhat distributed.
    # Just take first 26.
    centers_init = np.array(pts[:n])
    
    # 2. Optimization Loop
    # Variables: centers (n, 2), radii (n,)
    
    current_centers = centers_init.copy()
    current_radii = np.ones(n) * 0.01 # Small initial radii
    
    # Optimization parameters
    iterations = 1000
    step_size = 0.02
    
    for it in range(iterations):
        # Decay step size
        alpha = step_size * (1.0 - it / iterations)
        
        # --- Step A: Solve LP for Radii ---
        # Maximize sum(r_i)
        # Constraints:
        # 1. r_i + r_j <= dist(i, j) for all i < j
        # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # 3. r_i >= 0
        
        # Prepare LP
        # Objective: min -sum(r_i) => c = [-1, -1, ..., -1]
        c_obj = np.ones(n) * -1.0
        
        # Inequality constraints A_ub @ r <= b_ub
        A_ub = []
        b_ub = []
        
        # Pairwise constraints
        # To save memory, we can construct A_ub row by row or use sparse?
        # n=26, constraints ~ 325 + 104 = 430. Dense is fine.
        
        # We can compute distances once
        # dists matrix
        diffs = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Fill A_ub and b_ub
        # Pairs
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
                
            # Boundary constraints for circle i
            # r_i <= x_i
            row_x1 = np.zeros(n)
            row_x1[i] = 1.0
            A_ub.append(row_x1)
            b_ub.append(current_centers[i, 0])
            
            # r_i <= 1 - x_i
            row_x2 = np.zeros(n)
            row_x2[i] = 1.0
            A_ub.append(row_x2)
            b_ub.append(1.0 - current_centers[i, 0])
            
            # r_i <= y_i
            row_y1 = np.zeros(n)
            row_y1[i] = 1.0
            A_ub.append(row_y1)
            b_ub.append(current_centers[i, 1])
            
            # r_i <= 1 - y_i
            row_y2 = np.zeros(n)
            row_y2[i] = 1.0
            A_ub.append(row_y2)
            b_ub.append(1.0 - current_centers[i, 1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for r_i: (0, None)
        bounds = [(0, None) for _ in range(n)]
        
        # Solve
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            current_radii = res.x
        else:
            # Fallback if LP fails (shouldn't happen with small radii)
            current_radii = np.ones(n) * 0.001
            
        # --- Step B: Calculate Forces and Move Centers ---
        forces = np.zeros_like(current_centers)
        
        # Identify touching pairs (slack close to 0)
        # Slack = dist - (r_i + r_j)
        # If slack < tol, push apart.
        tol = 1e-6
        
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                r_sum = current_radii[i] + current_radii[j]
                slack = d - r_sum
                
                if slack < tol and d > 1e-9:
                    # Repulsion force
                    # Direction from j to i
                    vec = current_centers[i] - current_centers[j]
                    # Normalize
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        unit_vec = vec / norm
                        # Force magnitude
                        # Push proportional to radius sum? Or constant?
                        # Constant is safer to avoid oscillation
                        force_mag = 1.0 
                        forces[i] += force_mag * unit_vec
                        forces[j] -= force_mag * unit_vec
        
        # Boundary forces
        # If r_i > x_i - tol -> push right (increase x)
        # If r_i > 1-x_i - tol -> push left (decrease x)
        for i in range(n):
            x, y = current_centers[i]
            r = current_radii[i]
            
            # Left wall
            if r > x - tol:
                forces[i, 0] += 1.0
            # Right wall
            if r > (1.0 - x) - tol:
                forces[i, 0] -= 1.0
            # Bottom wall
            if r > y - tol:
                forces[i, 1] += 1.0
            # Top wall
            if r > (1.0 - y) - tol:
                forces[i, 1] -= 1.0
                
        # Update centers
        current_centers += alpha * forces
        
        # Clip to [0, 1]
        np.clip(current_centers, 0, 1, out=current_centers)
        
    # Final validation and cleanup
    # Run LP one last time to ensure radii are optimal for final centers
    # (Though the loop does this, this ensures consistency)
    
    # Re-solve LP for final configuration
    c_obj = np.ones(n) * -1.0
    A_ub = []
    b_ub = []
    
    diffs = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
        row_x1 = np.zeros(n); row_x1[i] = 1.0; A_ub.append(row_x1); b_ub.append(current_centers[i, 0])
        row_x2 = np.zeros(n); row_x2[i] = 1.0; A_ub.append(row_x2); b_ub.append(1.0 - current_centers[i, 0])
        row_y1 = np.zeros(n); row_y1[i] = 1.0; A_ub.append(row_y1); b_ub.append(current_centers[i, 1])
        row_y2 = np.zeros(n); row_y2[i] = 1.0; A_ub.append(row_y2); b_ub.append(1.0 - current_centers[i, 1])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        final_radii = res.x
    else:
        final_radii = np.ones(n) * 0.0 # Fallback
        
    sum_radii = np.sum(final_radii)
    
    return current_centers, final_radii, sum_radii
