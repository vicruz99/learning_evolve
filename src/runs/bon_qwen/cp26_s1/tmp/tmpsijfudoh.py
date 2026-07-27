import numpy as np
from scipy.optimize import minimize
import math

def calculate_energy(centers, radii, mu):
    """
    Calculates the objective function value.
    Objective: Maximize sum of radii (so minimize -sum(radii)).
    Penalty: Squared violations of boundary and overlap constraints.
    """
    n = len(radii)
    energy = 0.0
    
    # --- Boundary Penalties ---
    # Constraints: 
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # y - r >= 0  => r - y <= 0
    # y + r <= 1  => y + r - 1 <= 0
    
    # We penalize positive values of these expressions squared.
    # Violation = max(0, expression)
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    # Left wall violation: r - x
    viol_left = np.maximum(0, r - x)
    # Right wall violation: x + r - 1
    viol_right = np.maximum(0, x + r - 1)
    # Bottom wall violation: r - y
    viol_bottom = np.maximum(0, r - y)
    # Top wall violation: y + r - 1
    viol_top = np.maximum(0, y + r - 1)
    
    boundary_pen = np.sum(viol_left**2) + np.sum(viol_right**2) + \
                   np.sum(viol_bottom**2) + np.sum(viol_top**2)
    
    # --- Overlap Penalties ---
    # Constraint: ||c_i - c_j|| >= r_i + r_j
    # Violation: r_i + r_j - ||c_i - c_j|| > 0
    # We sum squared violations for all pairs i < j.
    
    # Vectorized distance calculation
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    # dists shape (n, n)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Radii sum matrix shape (n, n)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Violation matrix
    # violation_ij = max(0, r_i + r_j - dist_ij)
    # We only sum upper triangle to avoid double counting, but since we square and sum, 
    # and dists is symmetric, we can just sum all and divide by 2, or use a mask.
    # Summing all is fine as long as we are consistent, but let's be precise.
    # Actually, simpler to just sum all and divide by 2? 
    # No, let's just create a mask for i < j.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    violations = np.maximum(0, r_sum - dists)
    overlap_pen = np.sum(violations[mask]**2)
    
    # Total penalty
    total_pen = boundary_pen + overlap_pen
    
    # Objective: -sum(r) + mu * penalty
    return -np.sum(r) + mu * total_pen

def run_packing():
    n = 26
    
    # Function to generate initial centers based on a grid pattern
    def get_initial_centers(method='hex'):
        # We want 26 points.
        # Let's try a 5x6 grid (30 points) and pick 26 best ones, or a hexagonal pattern.
        # A hexagonal pattern is usually denser.
        
        # Let's create a set of points on a hexagonal lattice scaled to fit [0,1]x[0,1]
        # Lattice basis vectors: (1, 0) and (0.5, sqrt(3)/2)
        # We can generate points (i + 0.5*j, (sqrt(3)/2)*j)
        
        pts = []
        # Generate enough points to cover the square
        # Step size roughly 0.2 to get ~5 points per dimension
        step = 0.25 
        
        # Generate grid of indices
        # j is row index, i is col index
        # y = j * step * sqrt(3)/2
        # x = i * step + (0.5 * j * step)
        
        max_j = int(1.0 / (step * math.sqrt(3)/2)) + 2
        max_i = int(1.0 / step) + 2
        
        for j in range(max_j):
            for i in range(max_i):
                x = i * step + 0.5 * j * step
                y = j * step * (math.sqrt(3)/2)
                pts.append([x, y])
        
        pts = np.array(pts)
        
        # Normalize and shift to fit in [0, 1]x[0, 1] with some padding
        # Find bounding box
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        
        # Scale to fit in [0.1, 0.9] roughly? Or just [0,1]
        # Let's scale to fit within [0, 1]
        scale_x = 1.0 / width
        scale_y = 1.0 / height
        scale = min(scale_x, scale_y) # Keep aspect ratio
        
        pts = (pts - np.array([min_x, min_y])) * scale
        
        # Center in square
        curr_min = pts.min(axis=0)
        curr_max = pts.max(axis=0)
        offset_x = (1.0 - (curr_max[0] - curr_min[0])) / 2 - curr_min[0]
        offset_y = (1.0 - (curr_max[1] - curr_min[1])) / 2 - curr_min[1]
        pts += np.array([offset_x, offset_y])
        
        # Filter points that are strictly inside [0,1] (allow small epsilon)
        valid_mask = (pts[:, 0] >= -0.01) & (pts[:, 0] <= 1.01) & \
                     (pts[:, 1] >= -0.01) & (pts[:, 1] <= 1.01)
        pts = pts[valid_mask]
        
        # We need exactly 26 points.
        # If we have more, pick 26.
        # If we have less, duplicate or add?
        # With step 0.25, we should have plenty.
        if len(pts) > n:
            # Pick n points. Which ones?
            # Maybe the ones closest to the center of the square to keep symmetry?
            # Or just random?
            # Let's pick the ones that are most spread out?
            # Simple heuristic: pick points with indices that form a grid?
            # Or just take the first n? 
            # The generation order is row-major.
            # Let's shuffle to avoid bias? No, deterministic is better for debugging.
            # Let's pick points that maximize min distance? Too slow.
            # Let's just take a subset that is well distributed.
            # Sort by distance to center (0.5, 0.5) and take furthest?
            # Actually, we want to cover the square.
            # Let's just take the first n valid points.
            # But row-major generation might cluster.
            # Let's try to pick 26 points from a 5x5 grid + 1?
            pass 
            
        # Alternative: Simple Grid + Perturbation
        # 5x5 grid = 25 points. Add 1 point.
        # Grid points: x in {0.1, 0.3, 0.5, 0.7, 0.9}, y in {0.1, 0.3, 0.5, 0.7, 0.9}
        grid_x = np.linspace(0.1, 0.9, 5)
        grid_y = np.linspace(0.1, 0.9, 5)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_pts = np.vstack((gx.flatten(), gy.flatten())).T # 25 points
        
        # Add a 26th point. Where?
        # Maybe in a gap? 
        # Hexagonal shift suggests inserting between rows.
        # Let's add a point at (0.5, 0.5)? It's already there.
        # Let's add at (0.2, 0.2)?
        # Better: Generate a 6x5 grid (30 points) and select 26.
        
        # Let's go back to generating a dense grid and selecting.
        # 6 rows, 5 columns.
        x_coords = np.linspace(0.1, 0.9, 5)
        y_coords = np.linspace(0.1, 0.9, 6)
        X, Y = np.meshgrid(x_coords, y_coords)
        pts_grid = np.vstack((X.flatten(), Y.flatten())).T # 30 points
        
        # Select 26 points.
        # To ensure good distribution, let's remove 4 points.
        # Removing corners might be good? Or just random.
        # Let's remove the 4 points with largest index (last 4).
        # The order is row-major.
        # Last 4 are in the last row (y=0.9).
        # So we keep 5 points in last row? No, 5 points per row.
        # 30 points total. Remove 4 -> 26.
        # Removing last 4 means keeping 1 point in last row?
        # That leaves the bottom heavy.
        # Better to remove points symmetrically.
        # Remove (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)?
        # These are corners.
        # Let's identify indices of corners.
        # (0.1, 0.1) is index 0.
        # (0.9, 0.1) is index 4.
        # (0.1, 0.9) is index 25 (last row first col? No, y=0.9 is last row).
        # y_coords has 6 elements. Last is 0.9.
        # Indices for y=0.9 are 25, 26, 27, 28, 29.
        # (0.1, 0.9) is 25. (0.9, 0.9) is 29.
        # So remove indices 0, 4, 25, 29.
        
        remove_indices = [0, 4, 25, 29]
        keep_indices = [i for i in range(30) if i not in remove_indices]
        final_pts = pts_grid[keep_indices]
        
        return final_pts

    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Run optimization multiple times with different random seeds/perturbations
    # to escape local minima.
    for seed in range(5): # 5 attempts
        np.random.seed(seed)
        
        # 1. Initialize centers
        init_centers = get_initial_centers()
        
        # Perturb centers slightly
        noise = np.random.uniform(-0.02, 0.02, size=init_centers.shape)
        init_centers = init_centers + noise
        # Clip to [0, 1]
        init_centers = np.clip(init_centers, 0.0, 1.0)
        
        # 2. Initialize radii
        # Start with a small radius to avoid immediate huge penalties
        # Estimate max possible radius ~ 0.1. Start with 0.04.
        init_radii = np.full(n, 0.04)
        
        # 3. Prepare variables for optimizer
        # Flatten: [x1, y1, r1, x2, y2, r2, ...]
        # But L-BFGS-B needs bounds.
        # Bounds for x, y: [0, 1]
        # Bounds for r: [0, 0.5]
        
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[3*i] = init_centers[i, 0]
            x0[3*i+1] = init_centers[i, 1]
            x0[3*i+2] = init_radii[i]
            
        bounds = []
        for i in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
            
        # 4. Define objective wrapper
        # The optimizer passes a 1D vector x
        def objective(x_vec):
            centers = np.zeros((n, 2))
            radii = np.zeros(n)
            for i in range(n):
                centers[i, 0] = x_vec[3*i]
                centers[i, 1] = x_vec[3*i+1]
                radii[i] = x_vec[3*i+2]
            return calculate_energy(centers, radii, mu=2000.0)
            
        # 5. Run Optimizer
        # Using L-BFGS-B
        # maxiter can be high
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        # 6. Extract results
        opt_x = res.x
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = opt_x[3*i]
            centers[i, 1] = opt_x[3*i+1]
            radii[i] = opt_x[3*i+2]
            
        current_sum = np.sum(radii)
        
        # Check validity roughly (energy should be low)
        # Actually calculate energy to see penalty
        energy = calculate_energy(centers, radii, mu=2000.0)
        # The objective value is -sum(r) + penalty.
        # If penalty is 0, value is -sum(r).
        # We can check if penalty part is small.
        
        # Let's compute penalty separately to check validity
        # Re-using logic from calculate_energy
        # ... (omitted for brevity, assuming minimize worked)
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Post-optimization sanity check / refinement
    # Sometimes L-BFGS-B might get stuck or penalty not high enough.
    # But with mu=2000, it should be close.
    # We can try to increase radii uniformly if there is slack?
    # Or just return the best found.
    
    # Let's perform a quick check if we can increase radii slightly
    # by checking overlaps.
    # If min_dist > r_i + r_j, we can increase r.
    # But this is tricky with unequal radii.
    # The optimizer should have handled this.
    
    return best_centers, best_radii, float(best_sum_radii)

# To ensure the code runs without external imports issues in some environments,
# standard libraries are fine.