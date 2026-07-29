# sol_000196 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0dea8e43) state=88aef762 sum of radii=2.069086 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def get_initial_centers_hex(n_circles):
    """
    Generate initial centers using a hexagonal lattice pattern.
    Tries to form a compact shape.
    """
    centers = []
    # Hexagonal lattice basis vectors
    # We want distance between neighbors to be roughly uniform.
    # We will scale later, so let's assume unit distance for now.
    
    # Strategy: Fill rows. 
    # Estimate rows needed. Area ~ n * pi * (0.5)^2? 
    # Just try to fit in a hexagon or rectangle.
    
    # Let's try to generate points in a hex grid and pick closest to center?
    # Or just iterate rows.
    
    # Let's try a simple rectangular block of hex grid points.
    # 26 points.
    # Maybe 6 rows? 6 * 4 = 24, 6 * 5 = 30.
    # Try 5 rows? 5 * 5 = 25, 5 * 6 = 30.
    # Let's try 6 rows with 5, 4, 5, 4, 5, 3?
    
    # Actually, let's just generate a large grid and select 26 points closest to (0.5, 0.5)
    # after scaling to [0,1].
    
    # Generate grid
    pts = []
    # x spacing 1.0, y spacing sqrt(3)/2 approx 0.866? 
    # For hex packing, row offset is 0.5 * x_spacing.
    # Let's use spacing 1.0 for x.
    
    # Range of coordinates to cover unit square roughly
    for r in range(10): # rows
        for c in range(10): # cols
            x = c * 1.0 + (r % 2) * 0.5
            y = r * (np.sqrt(3) / 2.0)
            pts.append([x, y])
    
    pts = np.array(pts)
    
    # Shift and scale to fit in [0,1] roughly
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    
    # Normalize to [0, 1]
    pts[:, 0] = (pts[:, 0] - min_x) / (max_x - min_x)
    pts[:, 1] = (pts[:, 1] - min_y) / (max_y - min_y)
    
    # Select 26 points closest to center (0.5, 0.5)
    center = np.array([0.5, 0.5])
    dists = np.linalg.norm(pts - center, axis=1)
    indices = np.argsort(dists)[:n_circles]
    
    return pts[indices]

def objective_function(centers_flat, n_circles):
    """
    Objective function to maximize sum of radii.
    We calculate the maximum possible radius for each circle given the centers,
    assuming equal radii constraints locally (safe radii).
    Actually, to encourage valid packing, we compute r_i = min(dist_to_wall, min(dist_to_others)/2).
    This ensures r_i + r_j <= dist(i,j) is satisfied.
    We maximize sum(r_i).
    """
    centers = centers_flat.reshape(-1, 2)
    n = n_circles
    
    # Distance to walls
    dist_to_walls = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                               np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    # Distance to other centers
    # Compute pairwise distances
    # dist_matrix[i, j] = distance between i and j
    dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Min distance to any other circle
    min_dist_to_other = np.min(dist_matrix, axis=1)
    
    # Safe radius for each circle
    # r_i <= dist_to_wall_i
    # r_i <= min_dist_to_other_i / 2
    radii = np.minimum(dist_to_walls, min_dist_to_other / 2.0)
    
    # Clamp negative radii to 0 (should not happen if inside square)
    radii = np.maximum(radii, 0.0)
    
    return -np.sum(radii) # Minimize negative sum

def run_packing():
    n_circles = 26
    
    # 1. Generate initial centers
    centers_init = get_initial_centers_hex(n_circles)
    
    # 2. Optimize centers to maximize sum of safe radii
    # Using Nelder-Mead as it handles non-smooth objectives reasonably well
    # Initial guess flattened
    x0 = centers_init.flatten()
    
    # Bounds for centers [0, 1]
    bounds = [(0, 1)] * (2 * n_circles)
    
    # We use L-BFGS-B if we had gradients, but Nelder-Mead is derivative-free.
    # However, Nelder-Mead doesn't support bounds well directly in scipy?
    # Actually, simple bounds can be handled by penalty or transformation.
    # Let's use a simple coordinate scaling or just rely on the fact that optimal is interior.
    # Or use 'Powell' method.
    
    # To ensure centers stay in [0,1], we can use a penalty in objective or clip.
    # Clipping inside objective might break gradient logic but Nelder-Mead doesn't use gradients.
    # But better to transform variables?
    # Let's just run optimization and hope it stays in bounds.
    # If it goes out, the distance to wall becomes negative, radii become 0 (due to max with 0? no, min), sum drops.
    # Wait, dist_to_walls would be negative. min(negative, positive) = negative.
    # sum(radii) would be negative. Objective -sum would be positive.
    # We want to maximize sum, so minimize -sum.
    # If radii negative, sum negative, -sum positive.
    # Optimizer might seek negative radii?
    # We should enforce radii >= 0.
    
    # Let's rewrite objective to return a large penalty if centers out of bounds.
    
    def penalized_objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        # Check bounds
        if np.any(centers < -1e-5) or np.any(centers > 1 + 1e-5):
            return 1e6 # Penalty
        
        dist_to_walls = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                                   np.minimum(centers[:, 1], 1 - centers[:, 1]))
        
        # Pairwise distances
        # Efficient computation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dists = np.min(dists, axis=1)
        
        # Radii constrained by walls and neighbors (assuming equal radii logic for safety)
        # r_i <= dist_to_wall
        # r_i <= min_dist / 2
        radii = np.minimum(dist_to_walls, min_dists / 2.0)
        radii = np.maximum(radii, 0.0)
        
        return -np.sum(radii)

    # Optimization
    # Multiple restarts to avoid local minima
    best_res = None
    best_val = np.inf
    best_centers = None
    
    for restart in range(5):
        # Perturb initial centers slightly for each restart
        if restart == 0:
            x0_curr = centers_init.flatten()
        else:
            # Random perturbation
            noise = np.random.uniform(-0.05, 0.05, size=x0.shape)
            x0_curr = np.clip(centers_init.flatten() + noise, 0.01, 0.99)
        
        try:
            res = minimize(penalized_objective, x0_curr, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(-1, 2)
        except Exception:
            pass

    # 3. Refine Radii with Linear Programming
    # For the fixed centers, find optimal radii (allowing unequal) to maximize sum.
    # Constraints: r_i + r_j <= dist(i,j), r_i <= dist_to_wall(i), r_i >= 0.
    
    centers_opt = best_centers
    dist_to_walls = np.minimum(np.minimum(centers_opt[:, 0], 1 - centers_opt[:, 0]), 
                               np.minimum(centers_opt[:, 1], 1 - centers_opt[:, 1]))
    
    dist_matrix = np.sqrt(np.sum((centers_opt[:, np.newaxis, :] - centers_opt[np.newaxis, :, :]) ** 2, axis=2))
    
    # LP Setup
    # Maximize sum(r_i)  => Minimize -sum(r_i)
    c_obj = -np.ones(n_circles)
    
    # Inequalities A_ub @ r <= b_ub
    # 1. r_i + r_j <= dist(i,j) for i < j
    # 2. r_i <= dist_to_wall(i)
    
    n_constraints = n_circles * (n_circles - 1) // 2 + n_circles
    A_ub = np.zeros((n_constraints, n_circles))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    # Pairwise constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist_matrix[i, j]
            idx += 1
            
    # Wall constraints
    for i in range(n_circles):
        A_ub[idx, i] = 1.0
        b_ub[idx] = dist_to_walls[i]
        idx += 1
        
    # Bounds r_i >= 0
    bounds_r = [(0, None) for _ in range(n_circles)]
    
    # Solve LP
    # HighS is usually available in scipy, but method='highs' might not be in older versions.
    # Fallback to 'simplex' or 'interior-point'. 'highs' is preferred.
    try:
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    except:
        try:
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='interior-point')
        except:
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='simplex')
            
    if res_lp.success:
        radii_opt = res_lp.x
    else:
        # Fallback to equal radii estimation if LP fails
        min_dists = np.min(dist_matrix, axis=1)
        radii_opt = np.minimum(dist_to_walls, min_dists / 2.0)

    sum_radii = np.sum(radii_opt)
    
    # Final validation
    valid = validate_packing(centers_opt, radii_opt)
    if not valid:
        # If invalid, clamp radii slightly to fix?
        # Or just return what we have. The LP should guarantee validity.
        # But numerical errors might occur.
        pass
        
    return centers_opt, radii_opt, sum_radii

# Main execution wrapper
if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    print(f"Centers:\n{centers}")
    print(f"Radii:\n{radii}")
