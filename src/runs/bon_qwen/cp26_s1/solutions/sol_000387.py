# sol_000387 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 916b0b30) state=7aa9bcb6 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_overlap_loss(pos, radii):
    """
    Computes the sum of squared overlaps for circles with fixed radii
    and positions given by the flattened array pos.
    """
    N = len(radii)
    centers = pos.reshape(N, 2)
    loss = 0.0
    
    # Pairwise overlap penalty
    # Using a simple loop for clarity and correctness with small N
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                loss += overlap**2
                
    # Boundary overlap penalty
    for i in range(N):
        r = radii[i]
        x, y = centers[i]
        if x < r:
            loss += (r - x)**2
        elif x > 1 - r:
            loss += (x - (1 - r))**2
            
        if y < r:
            loss += (r - y)**2
        elif y > 1 - r:
            loss += (y - (1 - r))**2
            
    return loss

def run_packing():
    np.random.seed(42)
    N = 26
    
    # --- Initialization ---
    # Create a hexagonal grid pattern for 26 circles
    # Pattern: 5 rows with counts [5, 5, 5, 5, 6]
    row_counts = [5, 5, 5, 5, 6]
    centers = np.zeros((N, 2))
    idx = 0
    
    # Approximate initial radius to calculate spacing
    r_init = 0.08
    dx = 2 * r_init * 1.1  # Horizontal spacing
    dy = np.sqrt(3) * r_init * 1.1 # Vertical spacing
    
    for k, count in enumerate(row_counts):
        y = k * dy
        x_start = (k % 2) * (dx / 2) # Stagger odd rows
        for j in range(count):
            x = x_start + j * dx
            centers[idx] = [x, y]
            idx += 1
            
    # Normalize coordinates to fit within [0.1, 0.9] square
    min_c = centers.min()
    max_c = centers.max()
    if max_c > min_c:
        centers = (centers - min_c) / (max_c - min_c) * 0.8 + 0.1
    
    # Initial radii
    radii = np.full(N, 0.04)
    
    # Bounds for position optimization: circles must stay within [0, 1]
    # L-BFGS-B handles bounds efficiently
    pos_bounds = [(0.0, 1.0)] * (2 * N)
    
    # --- Optimization Loop ---
    num_iterations = 60
    
    for iteration in range(num_iterations):
        # Step 1: Optimize Radii using Linear Programming
        # Maximize sum(r_i) s.t. r_i + r_j <= dist(i,j) and boundary constraints
        c = np.ones(N) * -1 # Minimize -sum(r)
        
        A_ub = []
        b_ub = []
        
        # Pairwise constraints: r_i + r_j <= dist
        # Precompute distances for speed
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                dists[i, j] = d
                dists[j, i] = d
                
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
                
        # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
        for i in range(N):
            # r_i <= x_i
            row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(centers[i, 0])
            # r_i <= 1 - x_i
            row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(1 - centers[i, 0])
            # r_i <= y_i
            row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(centers[i, 1])
            # r_i <= 1 - y_i
            row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(1 - centers[i, 1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
            if res.success:
                radii = res.x
        except Exception:
            pass # Keep previous radii if LP fails
            
        # Step 2: Optimize Positions to reduce overlap
        # Minimize overlap loss given current radii
        # We pass radii as an argument to the objective function
        def obj_func(pos):
            return compute_overlap_loss(pos, radii)
            
        res_pos = minimize(obj_func, 
                           x0=centers.flatten(), 
                           method='L-BFGS-B', 
                           bounds=pos_bounds,
                           args=(), # radii is captured from scope, but let's be explicit if needed
                           options={'maxiter': 150, 'ftol': 1e-12})
                           
        if res_pos.success:
            centers = res_pos.x.reshape(N, 2)
            
    # Final LP step to ensure radii are exactly maximal for final positions
    c = np.ones(N) * -1
    A_ub = []
    b_ub = []
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(N); row[i] = 1; row[j] = 1
            A_ub.append(row); b_ub.append(d)
            
    for i in range(N):
        row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(centers[i, 0])
        row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(1 - centers[i, 0])
        row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(centers[i, 1])
        row = np.zeros(N); row[i] = 1; A_ub.append(row); b_ub.append(1 - centers[i, 1])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res_final = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if res_final.success:
        radii = res_final.x
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
