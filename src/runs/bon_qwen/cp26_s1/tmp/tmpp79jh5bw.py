import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Strategy:
    1. Initialize circles in a hexagonal lattice.
    2. Iteratively expand radii.
    3. Optimize centers to minimize overlap and boundary penalties.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We try to fit 26 circles in a hexagonal pattern.
    # Rows of length ~5 or 6.
    centers = []
    # Approximate radius for 26 circles ~ 0.1
    # Grid spacing dx = 2*r, dy = sqrt(3)*r
    # Let's just place them in a dense grid first and let optimization fix it.
    # A 6x5 grid has 30 spots, we pick 26.
    # Or simply a dense random init with some spread.
    
    # Let's use a structured init: 5 rows of 5, 1 extra?
    # Or just a dense grid.
    # Let's try to distribute them evenly in [0,1]x[0,1]
    x_coords = np.linspace(0.1, 0.9, 6) # 6 points
    y_coords = np.linspace(0.1, 0.9, 5) # 5 points
    # 30 points. We need 26.
    # Let's pick a subset or just use a random perturbation of a grid.
    
    # Better init: Hexagonal packing
    r_init = 0.1
    centers = []
    count = 0
    row_y = r_init
    while count < n:
        # Determine row length
        # Width 1, spacing 2r
        max_circles_row = int((1 - 2*r_init) / (2*r_init)) + 1
        # Actually for hex, row 1: 5, row 2: 4, row 3: 5...
        # Let's just fill rows
        row_len = min(5, n - count) 
        if len(centers) > 0 and (len(centers) // 5) % 2 == 1:
             row_len = min(4, n - count) # Alternating 5,4,5,4...
        
        # If we are running out, just put remaining
        if n - count < row_len:
            row_len = n - count
            
        y = row_y
        for i in range(row_len):
            if count >= n: break
            x = r_init + i * (2 * r_init)
            centers.append([x, y])
            count += 1
        row_y += r_init * np.sqrt(3)
        if row_y + r_init > 1:
            # Wrap around or just break? 
            # If we break, we might not have 26.
            # Let's reset y if too high? No, just fill as much as possible.
            # But we need 26.
            pass 

    # If the logic above failed to generate 26, fallback to random/grid
    if len(centers) < n:
        centers = []
        # Create a grid
        # sqrt(26) ~ 5.1
        # 6x5 grid
        xs = np.linspace(0.1, 0.9, 6)
        ys = np.linspace(0.1, 0.9, 5)
        for y in ys:
            for x in xs:
                if len(centers) < n:
                    centers.append([x, y])
    
    centers = np.array(centers[:n])
    radii = np.full(n, 0.01) # Start small

    # 2. Iterative Optimization
    # We will try to increase radii and optimize positions
    
    # Define penalty function
    def objective(params):
        # params: flattened [x0, y0, r0, x1, y1, r1, ...]
        # But we will optimize centers only, keeping radii fixed in the loop?
        # Actually, optimizing both is hard. 
        # Strategy: Fix radii, optimize centers to minimize overlap.
        # Then increase radii.
        pass

    # Let's use a loop:
    # 1. Fix radii.
    # 2. Optimize centers to minimize penalty.
    # 3. Increase radii.
    
    # Penalty for fixed radii:
    # Sum of max(0, r_i + r_j - dist_ij)^2 + boundary penalties
    
    def get_penalty(centers, radii):
        penalty = 0.0
        n = len(radii)
        # Pairwise overlaps
        # Vectorized computation for speed
        # dist matrix
        # diff = centers[:, None, :] - centers[None, :, :] # (26, 26, 2)
        # dist = np.linalg.norm(diff, axis=2) # (26, 26)
        # To avoid diagonal and double counting
        # upper triangle
        for i in range(n):
            r_i = radii[i]
            c_i = centers[i]
            # Boundary penalties
            # x - r < 0 => r - x > 0
            p_x_neg = max(0, r_i - c_i[0])
            p_x_pos = max(0, c_i[0] - (1 - r_i))
            p_y_neg = max(0, r_i - c_i[1])
            p_y_pos = max(0, c_i[1] - (1 - r_i))
            penalty += (p_x_neg**2 + p_x_pos**2 + p_y_neg**2 + p_y_pos**2) * 1000 # High weight
            
            for j in range(i + 1, n):
                r_j = radii[j]
                c_j = centers[j]
                dist = np.sqrt((c_i[0]-c_j[0])**2 + (c_i[1]-c_j[1])**2)
                overlap = r_i + r_j - dist
                if overlap > 0:
                    penalty += overlap**2 * 100 # High weight
        return penalty

    # Optimizer function for centers
    def optimize_centers(centers, radii):
        # Flatten centers for scipy
        x0 = centers.flatten()
        
        def func(x_flat):
            c = x_flat.reshape(n, 2)
            return get_penalty(c, radii)
        
        # Bounds for centers [0, 1]
        bounds = [(0, 1)] * (2 * n)
        
        # Use L-BFGS-B
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
        return res.x.reshape(n, 2)

    # Expansion loop
    radii = np.full(n, 0.01)
    # Initialize centers properly first
    # Run one optimization with small radii to center them
    centers = optimize_centers(centers, radii)
    
    step = 0.0005
    max_iter = 2000 # Limit iterations
    
    for _ in range(max_iter):
        # Try to increase radii
        # Check if valid
        pen = get_penalty(centers, radii)
        if pen < 1e-6:
            # Valid, expand
            radii += step
            centers = optimize_centers(centers, radii)
        else:
            # Not valid, maybe reduce step or just try to optimize harder
            # For simplicity, if penalty high, maybe we are stuck.
            # But let's just continue optimizing centers with current radii
            centers = optimize_centers(centers, radii)
            # If still invalid after optimization, reduce radii slightly to recover?
            # Or just keep current radii.
            # Let's reduce radii to ensure we can proceed
            if get_penalty(centers, radii) > 1e-4:
                radii -= step * 2
                continue
            else:
                radii += step * 0.1 # Small expansion
    
    # Final optimization to clean up
    centers = optimize_centers(centers, radii)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii