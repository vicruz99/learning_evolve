# sol_000177 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b137705a) state=36dc5a45 sum of radii=1.620127 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Initialization: Hexagonal Lattice ---
    # We try to fit 26 circles in a hexagonal pattern.
    # A 5x5 grid fits 25. We need 1 more.
    # A hexagonal packing might allow better density.
    # Let's try to arrange them in rows.
    # Rows of 5, 5, 5, 5, 4, 2? Or 5, 6, 5, 5, 5?
    # Let's just generate a dense random-ish or grid pattern and let optimizer fix it.
    # A simple grid is a safe start.
    
    # Let's try a perturbed 5x5 grid + 1 in center
    centers = np.zeros((n, 2))
    
    # 5x5 grid for first 25
    count = 0
    step = 1.0 / 6.0 # slightly smaller than 1/5 to leave room
    start = 0.1 + 0.02 # start a bit in
    
    # Better initialization: Hexagonal packing
    # Estimate radius r ~ 0.1
    # Vertical spacing r*sqrt(3) ~ 0.1732
    # Horizontal spacing 2r ~ 0.2
    
    # Let's place them roughly
    # 6 rows?
    # Row 0: 4 circles
    # Row 1: 5 circles
    # Row 2: 4 circles
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Row 5: 4 circles
    # Total 26? 4+5+4+5+4+4 = 26.
    
    r_est = 0.09
    y_step = r_est * np.sqrt(3)
    
    row_configs = [
        (4, 0.0),      # Row 0, 4 circles, shift 0
        (5, 0.5),      # Row 1, 5 circles, shift 0.5 (relative to step?)
        (4, 0.0),
        (5, 0.5),
        (4, 0.0),
        (4, 0.0)
    ]
    # Adjust shifts: In hexagonal, rows alternate shift by r_est.
    # If row 0 starts at x=r_est, row 1 starts at x=2*r_est.
    
    current_y = r_est
    idx = 0
    centers = np.zeros((n, 2))
    
    # Let's just place them in a compact hexagonal grid
    # We will scale the whole configuration to fit later if needed, 
    # but for initialization, valid positions are good.
    
    # Simple approach: Place in a grid and let optimizer move them
    # Grid 5x5 is 25.
    # Let's use a dense packing heuristic initialization.
    
    # Try to fit 26 circles with r=0.08
    # Width 1.0. 1/0.16 = 6.25. So 6 circles fit in a row with r=0.08?
    # 6 * 0.16 = 0.96. Yes.
    # So we can have rows of 6.
    # 26 circles -> 5 rows of 6? No 5*6=30.
    # 4 rows of 6 = 24. Need 2 more.
    # Rows: 6, 6, 6, 6, 2?
    # Vertical: 5 rows. Height 2r + 4*r*sqrt(3) = 0.16 + 0.32*1.732 = 0.16 + 0.55 = 0.71. Fits.
    
    # Let's construct this initialization
    r_init = 0.08
    dy = r_init * np.sqrt(3)
    dx = 2 * r_init
    
    y_current = r_init
    idx = 0
    
    # Row patterns: 6, 5, 6, 5, 4 (Sum 26)
    # Alternating shifts
    row_counts = [6, 5, 6, 5, 4]
    
    for i, count in enumerate(row_counts):
        # Shift: if i is even, start at r_init. If odd, start at 2*r_init
        if i % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
            
        for j in range(count):
            if idx < n:
                x = x_start + j * dx
                y = y_current
                centers[idx] = [x, y]
                idx += 1
        
        y_current += dy

    # --- Optimization ---
    
    # Define function to compute max radii for given centers
    def get_max_radii(centers_flat):
        c = centers_flat.reshape(n, 2)
        r = np.full(n, 0.0)
        
        # Compute distance matrix
        # dist[i, j] = distance between i and j
        # We can compute this on the fly or precompute
        # Since we call this inside optimize, precomputing inside might be costly
        # But n=26 is small.
        
        # Iterative relaxation for radii
        # r_i <= dist(c_i, c_j) - r_j
        # r_i <= boundary distances
        
        bounds = np.min(np.column_stack([
            c[:, 0], 
            1.0 - c[:, 0], 
            c[:, 1], 
            1.0 - c[:, 1]
        ]), axis=1)
        
        # Distance matrix
        # shape (n, n)
        dist = np.sqrt(((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Diagonal is 0, we don't care
        np.fill_diagonal(dist, np.inf)
        
        # Iterate to find max radii
        # Start with r=0
        for _ in range(50): # 50 iterations should be enough for n=26
            # r_i_new = min(bound_i, min_j(dist_ij - r_j))
            # We can vectorize:
            # For each i, we want min over j of (dist[i,j] - r[j])
            # dist matrix is (n,n), r is (n,)
            # dist[i,:] - r[:] gives array of size n
            # min over axis 1
            new_r = np.minimum(bounds, np.min(dist - r, axis=1))
            # Ensure non-negative
            new_r = np.maximum(new_r, 0.0)
            # Check convergence
            if np.allclose(r, new_r, atol=1e-9):
                break
            r = new_r
            
        return r

    # Objective function: Minimize negative sum of radii
    def objective(centers_flat):
        # Check for invalid centers (outside box) - penalty
        c = centers_flat.reshape(n, 2)
        # Soft constraint penalty to keep centers in [0,1]
        # Though Nelder-Mead doesn't support bounds easily, we can clamp or penalize
        # Better to clamp to [0,1]
        c = np.clip(c, 0.0, 1.0)
        
        radii = get_max_radii(c.flatten())
        return -np.sum(radii)

    # Run optimization
    # Nelder-Mead is robust
    # We flatten the centers array
    x0 = centers.flatten()
    
    # To improve results, we can run a few restarts or just one long run
    # With 52 variables, Nelder-Mead might take many function evaluations.
    # Each evaluation does 50 iterations of 26x26 matrix ops. 26^2 is small (676).
    # So it should be fast.
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-9})
    
    # Extract optimized centers
    best_centers = result.x.reshape(n, 2)
    # Compute final radii
    best_radii = get_max_radii(best_centers.flatten())
    
    # Final validation and return
    # Ensure centers are strictly inside [0,1] if radii are large
    # The get_max_radii function handles boundary constraints by limiting radii,
    # but centers themselves should be valid.
    # Clipping centers might have happened, which is fine.
    
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, float(sum_radii)

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Optional: Validate
    # import sys
    # sys.path.append('path/to/validation') # assuming validate_packing is available
    # print(validate_packing(centers, radii))
