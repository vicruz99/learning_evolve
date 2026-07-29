# sol_000212 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8fb167b6) state=de779b1b sum of radii=2.515389 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initialization ---
    # Generate a hexagonal lattice of points inside the unit square.
    # Hexagonal packing is denser than square grid, providing a better initial guess.
    initial_radii = 0.05
    spacing = 2 * initial_radii
    dy = np.sqrt(3) * spacing
    
    centers_list = []
    y = initial_radii
    row = 0
    
    # Generate rows until we exceed y=1
    while y + initial_radii <= 1.0:
        x = initial_radii
        # Offset odd rows by radius amount to form hexagonal pattern
        if row % 2 == 1:
            x += initial_radii
            
        # Add points in the row
        while x + initial_radii <= 1.0:
            centers_list.append([x, y])
            x += spacing
        
        y += dy
        row += 1
    
    # If we generated fewer than 26 points (unlikely with 0.05 radius), fill with random
    while len(centers_list) < n:
        centers_list.append([np.random.rand(), np.random.rand()])
    
    # Take exactly 26 points
    init_centers = np.array(centers_list[:n])
    init_radii = np.full(n, initial_radii)
    
    # Flatten variables: [x1, y1, x2, y2, ..., r1, r2, ...]
    # Total 3*n variables. First 2*n are centers, last n are radii.
    x0 = np.concatenate([init_centers.flatten(), init_radii])
    
    # --- 2. Objective and Constraints ---
    
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(vars[2*n:])
    
    def constraints(vars):
        centers = vars[:2*n].reshape(n, 2)
        radii = vars[2*n:]
        
        constraints_list = []
        
        # Boundary constraints: r <= x <= 1-r  and  r <= y <= 1-r
        # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        constraints_list.append(centers[:, 0] - radii)
        constraints_list.append(1.0 - centers[:, 0] - radii)
        constraints_list.append(centers[:, 1] - radii)
        constraints_list.append(1.0 - centers[:, 1] - radii)
        
        # Non-overlap constraints: ||c_i - c_j|| >= r_i + r_j
        # Vectorized calculation of distance matrix
        # diff shape: (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Radii sum matrix
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Extract upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices(n, k=1)
        dist_upper = dist[upper_tri_indices]
        radii_sum_upper = radii_sum[upper_tri_indices]
        
        constraints_list.append(dist_upper - radii_sum_upper)
        
        return np.concatenate(constraints_list)
    
    # --- 3. Optimization ---
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    # Centers (first 2n vars) bounded by [0, 1]
    # Radii (last n vars) bounded by [0, 0.5]
    bnds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    # Setup constraint dictionary for SLSQP
    cons = ({'type': 'ineq', 'fun': constraints})
    
    # Run optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bnds, 
        constraints=cons, 
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # --- 4. Post-processing ---
    # Extract results
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # Numerical cleanup: clamp radii to non-negative and ensure valid bounds
    # If a radius is slightly negative due to numerical error, fix it.
    final_radii = np.maximum(final_radii, 0.0)
    
    # Ensure centers are within [0, 1] and circles inside square
    for i in range(n):
        r = final_radii[i]
        final_centers[i, 0] = np.clip(final_centers[i, 0], r, 1.0 - r)
        final_centers[i, 1] = np.clip(final_centers[i, 1], r, 1.0 - r)
        
    # Final validation check and repair if necessary
    # If constraints were violated slightly, shrink radii
    # This is a safety net, though SLSQP should satisfy constraints.
    for i in range(n):
        r_i = final_radii[i]
        # Check boundaries
        max_r_bound = min(final_centers[i, 0], 1.0 - final_centers[i, 0], 
                          final_centers[i, 1], 1.0 - final_centers[i, 1])
        r_i = min(r_i, max(0, max_r_bound))
        
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j]) ** 2))
            r_j = final_radii[j]
            # If overlap, reduce radii proportionally or just shrink
            if dist < r_i + r_j - 1e-9:
                overlap = r_i + r_j - dist
                # Reduce both slightly to resolve
                reduction = overlap / 2 + 1e-9
                final_radii[i] -= reduction
                final_radii[j] -= reduction
                final_radii[i] = max(0, final_radii[i])
                final_radii[j] = max(0, final_radii[j])
                
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, float(sum_radii)

# For local testing
if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")
    print(f"Centers shape: {centers.shape}, Radii shape: {radii.shape}")
    
    # Run validation
    try:
        import sys
        # Assuming validate_packing is available in the environment
        # or we can just print a simple check
        valid = True
        for i in range(len(radii)):
            x, y = centers[i]
            r = radii[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                valid = False
                break
            for j in range(i+1, len(radii)):
                dist = np.sqrt((x-centers[j,0])**2 + (y-centers[j,1])**2)
                if dist < r + radii[j] - 1e-12:
                    valid = False
                    break
        print(f"Manual Validation: {valid}")
    except Exception as e:
        print(f"Error: {e}")
