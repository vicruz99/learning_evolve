# sol_000336 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=c57925e6 sum of radii=2.161965 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_grid(n):
    """
    Generate an initial configuration of n circles using a hexagonal pattern.
    """
    centers = []
    r_est = 0.1 # Initial estimate
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    row = 0
    col = 0
    while len(centers) < n:
        x = col * dx
        y = row * dy
        # Stagger odd rows
        if row % 2 == 1:
            x += dx / 2.0
        
        # Check if point fits roughly in [0, 1]
        if x < 1.0 and y < 1.0:
            centers.append([x, y])
        
        col += 1
        if x + dx > 1.0: # End of row logic approximation
             col = 0
             row += 1
             if row % 2 == 0:
                 col = 1 # Start next row offset if needed, but logic handled by loop

    # Trim or pad if necessary, though loop should handle count
    centers = centers[:n]
    return np.array(centers)

def objective_function(params, n):
    """
    Objective function to minimize.
    We want to maximize radii, which is equivalent to minimizing a penalty for 
    constraint violations (overlap and boundary exit).
    
    Params layout: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    centers = params[0:2*n].reshape((n, 2))
    radii = params[2*n:]
    
    penalty = 0.0
    
    # Boundary constraints: center must be at least r away from walls
    # x - r >= 0 => r - x <= 0
    # x + r <= 1 => x + r - 1 <= 0
    # Similarly for y
    
    # We can enforce radii to be equal or allow them to vary. 
    # For sum of radii, varying is allowed, but equal is often optimal for packing.
    # Let's enforce equality to simplify optimization landscape, or just penalize differences?
    # Actually, let's just optimize positions and a single radius r.
    # But the params vector includes radii. Let's enforce r_i = r_common.
    # Or better, just optimize positions and scale radii?
    # Let's stick to optimizing positions and radii, but add a term to encourage equal radii
    # or just return -sum(radii) if constraints satisfied.
    
    # Let's try a constrained optimization approach using penalty method.
    # Objective: - sum(radii) + penalties
    
    # 1. Sum of radii (we want to maximize, so minimize negative)
    obj = -np.sum(radii)
    
    # 2. Overlap penalties
    # dist_ij >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap**2
            
            # Boundary penalties
            # Circle i
            for coord in range(2):
                val = centers[i, coord]
                r = radii[i]
                # Lower bound
                if val - r < 0:
                    penalty += (val - r)**2
                # Upper bound
                if val + r > 1:
                    penalty += (val + r - 1)**2

    return obj + penalty * 1000 # High weight on constraints

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers and radii
    # Using a hexagonal layout
    initial_centers = generate_hexagonal_grid(n)
    
    # Normalize initial centers to be within [0, 1] if they went out
    min_c = np.min(initial_centers, axis=0)
    max_c = np.max(initial_centers, axis=0)
    scale = 0.9 / (max_c - min_c) # Scale to fit with some margin
    initial_centers = (initial_centers - min_c) * scale + (1 - scale) / 2
    
    # Initial radii: start small to avoid immediate penalty
    initial_radii = np.full(n, 0.05)
    
    # Combine into params
    x0 = np.concatenate([initial_centers.flatten(), initial_radii])
    
    # Bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1)]) # x, y
        bounds.append((0, 0.5)) # radius
    
    # Optimization
    # Using L-BFGS-B for bounded variables
    res = minimize(objective_function, x0, args=(n,), method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 5000, 'ftol': 1e-9, 'gtol': 1e-9})
    
    opt_params = res.x
    centers = opt_params[0:2*n].reshape((n, 2))
    radii = opt_params[2*n:]
    
    # Post-processing:
    # The optimization might result in slightly unequal radii or small violations.
    # We should clamp radii to ensure validity.
    # Find the maximum valid radius such that no overlap/boundary violation occurs.
    # This is a safe fallback to ensure validity, though the optimizer should be close.
    
    # Check validity and adjust if needed
    # Simple iterative shrinking if overlap exists
    for _ in range(100):
        overlap_exists = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j]:
                    # Shrink both
                    excess = radii[i] + radii[j] - dist
                    radii[i] -= excess / 2
                    radii[j] -= excess / 2
                    overlap_exists = True
            
            # Boundary check
            for d in range(2):
                if centers[i, d] - radii[i] < 0:
                    radii[i] = centers[i, d]
                    overlap_exists = True
                if centers[i, d] + radii[i] > 1:
                    radii[i] = 1 - centers[i, d]
                    overlap_exists = True
        
        if not overlap_exists:
            break
            
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
