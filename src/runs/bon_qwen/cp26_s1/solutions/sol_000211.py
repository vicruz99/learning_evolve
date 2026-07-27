# sol_000211 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cccf4974) state=a5520767 sum of radii=2.383341 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26)
        sum_radii: float
    """
    
    def compute_radii(centers):
        """
        Compute the maximum valid radii for a given set of centers.
        r_i is limited by distance to boundary and half distance to nearest neighbor.
        """
        n = centers.shape[0]
        radii = np.zeros(n)
        
        # Distance to boundary
        x = centers[:, 0]
        y = centers[:, 1]
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        radii = np.maximum(radii, dist_boundary) # Start with boundary limit

        # Distance to neighbors
        # Compute pairwise distances
        # Using broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # We only care about j != i. Set diagonal to infinity so min ignores it.
        np.fill_diagonal(dists, np.inf)
        
        # Half distance to nearest neighbor
        min_dist_to_neighbor = np.min(dists, axis=1)
        radii_from_neighbors = min_dist_to_neighbor / 2.0
        
        # The radius is the minimum of boundary limit and neighbor limit
        radii = np.minimum(radii, radii_from_neighbors)
        
        return radii

    def objective_func(centers_flat):
        """
        Objective function to maximize: sum of radii.
        Optimizers minimize, so we return negative sum.
        """
        n = len(centers_flat) // 2
        centers = centers_flat.reshape((n, 2))
        radii = compute_radii(centers)
        return -np.sum(radii)

    n_circles = 26
    best_centers = None
    best_sum_radii = -np.inf
    
    # Strategy: Try multiple initial configurations
    # 1. Hexagonal packing
    # 2. Random perturbations of hexagonal packing
    # 3. Pure random (less likely to be good, but safe)
    
    def get_hexagonal_init():
        # Approximate hexagonal packing
        # Try to fit rows
        # Spacing dx = 2r, dy = r*sqrt(3)
        # We don't know r yet, but we can place points relative to each other.
        # Let's place them in a grid first and then optimize.
        # A 5x5 grid has 25 points. We need 26.
        # Maybe 6 rows?
        # Let's just generate a dense random set and let optimizer work,
        # but structured is better.
        
        # Let's try a simple grid and add one point
        centers = np.zeros((n_circles, 2))
        
        # 5x5 grid would be ideal for 25. For 26, maybe 5x5 plus one?
        # Or a hexagonal arrangement.
        
        # Hexagonal lattice generation
        # Points at (i*dx + (j%2)*dx/2, j*dy)
        # Let's estimate r ~ 0.1. dx = 0.2, dy = 0.1732
        dx = 0.2
        dy = 0.1732
        
        pts = []
        # Try filling rows
        for j in range(6): # 6 rows
            for i in range(5): # 5 cols
                if len(pts) >= 26:
                    break
                x = i * dx + (j % 2) * (dx / 2) + 0.1 # Offset
                y = j * dy + 0.1
                # Keep inside [0,1]
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
            if len(pts) >= 26:
                break
        
        if len(pts) < 26:
            # Fill with random if not enough
            while len(pts) < 26:
                pts.append([np.random.rand(), np.random.rand()])
        
        return np.array(pts[:26])

    def get_grid_init():
        centers = np.zeros((n_circles, 2))
        # 5x5 grid for 25, plus one in center or corner?
        # Just spread them out
        cols = 5
        rows = 6 # 30 slots
        # Pick 26
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < 26:
                    # Map to [0,1]
                    # Leave some margin
                    margin = 0.05
                    x = margin + c * (1 - 2*margin) / (cols - 1)
                    y = margin + r * (1 - 2*margin) / (rows - 1)
                    centers[idx] = [x, y]
                    idx += 1
        return centers

    initial_configs = []
    
    # Config 1: Hexagonal
    initial_configs.append(get_hexagonal_init())
    
    # Config 2: Grid
    initial_configs.append(get_grid_init())
    
    # Config 3-6: Perturbed Hexagonal
    base_hex = get_hexagonal_init()
    for _ in range(4):
        noise = np.random.randn(26, 2) * 0.05
        perturbed = base_hex + noise
        # Clip to bounds
        perturbed = np.clip(perturbed, 0, 1)
        initial_configs.append(perturbed)

    for init_centers in initial_configs:
        # Reshape to 1D for optimizer
        x0 = init_centers.flatten()
        
        # Bounds for centers [0, 1]
        bounds = [(0, 1)] * (26 * 2)
        
        # Use Nelder-Mead as it doesn't require gradients and handles non-smooth objectives reasonably well
        # Or L-BFGS-B with numerical gradient approximation (method='L-BFGS-B' handles bounds well)
        # Nelder-Mead is robust but slow in high dimensions. 
        # Let's try 'Powell' or 'Nelder-Mead'.
        
        # Using SLSQP with numerical gradient might be faster but tricky with non-smooth min.
        # Let's use Nelder-Mead with a limit on iterations.
        
        res = opt.minimize(objective_func, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'adaptive': True})
        
        centers_opt = res.x.reshape((n_circles, 2))
        radii_opt = compute_radii(centers_opt)
        current_sum = np.sum(radii_opt)
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers_opt.copy()

    # Final validation and cleanup
    # Ensure centers are strictly inside and radii are valid
    # The compute_radii function ensures r_i <= dist to boundary and r_i <= dist/2 to neighbors.
    # So overlaps are impossible by construction (dist >= 2*r_i? No. dist >= r_i + r_j?
    # Wait. r_i = min(dist(i,j)/2). So 2*r_i <= dist(i,j).
    # Also r_j <= dist(i,j)/2.
    # So r_i + r_j <= dist(i,j)/2 + dist(i,j)/2 = dist(i,j).
    # So no overlap.
    
    # However, we need to make sure radii are non-negative.
    radii_final = compute_radii(best_centers)
    
    # Small epsilon correction to ensure strict inequality if needed by validator?
    # Validator allows 1e-12 error.
    # But let's just return the computed values.
    
    return best_centers, radii_final, np.sum(radii_final)

# Note: In a real execution environment, we would call run_packing().
# Here we just define it.
