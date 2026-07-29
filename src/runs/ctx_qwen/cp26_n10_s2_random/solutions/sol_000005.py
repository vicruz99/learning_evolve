# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2a4ed9f3) state=8a02f396 sum of radii=2.201036 correctness=1.0
# stdout(first 200): Starting global optimization (Differential Evolution)... DE Optimization finished. Best value: 1.6987642324269951 Starting local refinement (Nelder-Mead)... Final sum of radii: 2.2010360495967074
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution, minimize

def get_max_radii(centers):
    """
    Computes the maximum valid radius for each circle given fixed centers.
    r_i = min(dist to boundary, 0.5 * min(dist to other centers))
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Precompute boundary distances
    # Distance to left, right, bottom, top
    dist_left = centers[:, 0]
    dist_right = 1.0 - centers[:, 0]
    dist_bottom = centers[:, 1]
    dist_top = 1.0 - centers[:, 1]
    boundary_dists = np.minimum(np.minimum(dist_left, dist_right), 
                                np.minimum(dist_bottom, dist_top))
    
    # Compute pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # For each i, find min distance to j != i
    # We can set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pairwise_dists = np.min(dists, axis=1)
    
    # Radius is limited by half the distance to nearest neighbor
    circle_dists = 0.5 * min_pairwise_dists
    
    # Radius is min of boundary limit and neighbor limit
    radii = np.minimum(boundary_dists, circle_dists)
    
    return radii

def objective(centers_flat):
    """
    Objective function for optimization.
    Minimizes negative sum of radii.
    """
    centers = centers_flat.reshape(-1, 2)
    # Clip centers to [0, 1] to handle boundary constraints in optimizer if needed,
    # though bounds are handled by DE.
    centers = np.clip(centers, 1e-9, 1 - 1e-9)
    
    radii = get_max_radii(centers)
    return -np.sum(radii)

def initialize_centers_hexagonal(n):
    """
    Generates an initial set of centers using a hexagonal packing pattern.
    """
    centers = []
    # Estimate grid spacing
    # For n=26, roughly 5x5 or hexagonal.
    # Let's try to fit points in a hexagonal grid
    
    # Heuristic for grid size
    # If we have k rows, and approx n/k points per row.
    # Spacing s. 
    # Let's just generate a grid and pick best n points or similar.
    # Or simpler: place points in rows.
    
    # Try to determine number of rows
    # For hex packing, area per circle ~ sqrt(3)/2 * (2r)^2 = 2*sqrt(3)*r^2.
    # Total area 1. n * 2*sqrt(3)*r^2 ~ 1 => r ~ sqrt(1 / (2*sqrt(3)*n))
    # r ~ 0.08 for n=26. Spacing 2r ~ 0.16.
    # 1/0.16 ~ 6.
    
    spacing = 0.16
    centers_list = []
    
    y = spacing / 2  # Start with margin
    while y < 1 - spacing / 2:
        x = spacing / 2
        if len(centers_list) % 2 == 1:
            x = spacing  # Offset for hexagonal
        while x < 1 - spacing / 2:
            centers_list.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        
    # If we have fewer or more points, we can adjust.
    # But this is just an initialization.
    # We need exactly 26 points.
    
    # If we have more, pick first 26. If fewer, add random or extend.
    centers_arr = np.array(centers_list)
    
    if len(centers_arr) > n:
        # Take a subset, maybe centered?
        # Or just take first n.
        # To be safe, let's take the ones most centrally located or just first n.
        # Random selection might be better to avoid bias, but deterministic is fine for init.
        # Let's take indices 0 to n-1.
        return centers_arr[:n]
    else:
        # Need more points. Add random points in valid regions.
        remaining = n - len(centers_arr)
        # Simple rejection sampling to add points
        new_pts = []
        attempts = 0
        while len(new_pts) < remaining and attempts < 1000:
            pt = np.random.rand(2)
            # Check distance to existing points
            dists = np.linalg.norm(centers_arr - pt, axis=1)
            if np.min(dists) > spacing * 0.5: # Keep some separation
                new_pts.append(pt)
            attempts += 1
        if len(new_pts) > 0:
            return np.vstack([centers_arr, np.array(new_pts)])
        else:
            # Fallback: random points
            return np.random.rand(n, 2)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions of 26 circles to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialize centers
    # Use a deterministic seed for reproducibility if needed, but random is okay.
    # Let's use a hexagonal initialization for a good start.
    centers_init = initialize_centers_hexagonal(n)
    
    # Ensure centers are within [0,1]
    centers_init = np.clip(centers_init, 0.01, 0.99)
    
    # 2. Define bounds for optimization
    # Each center (x, y) is in [0, 1]
    bounds = [(0, 1) for _ in range(n * 2)]
    
    # 3. Run Differential Evolution
    # DE is global but can be slow. We use maxiter and popsize to control.
    # We optimize the negative sum of radii.
    # Note: DE expects the function to minimize.
    
    # We can try a few random restarts if needed, but DE has a population.
    # Let's run DE.
    # Since the objective function has discontinuities (min function), 
    # DE is suitable.
    
    print("Starting global optimization (Differential Evolution)...")
    # Adjust parameters for speed vs accuracy
    # population_size = n*20 = 520 might be heavy. 
    # Let's try smaller.
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=200, 
        popsize=20, 
        mutation=(0.5, 1), 
        recombination=0.7,
        tol=1e-7
    )
    
    print(f"DE Optimization finished. Best value: {-result_de.fun}")
    
    best_centers_flat = result_de.x
    best_centers = best_centers_flat.reshape(-1, 2)
    
    # 4. Local Refinement
    # Use L-BFGS-B to polish the solution.
    # L-BFGS-B requires gradients or finite differences. 
    # Our objective is non-smooth, so we might need 'Nelder-Mead' or 'Powell' which don't require gradients.
    # But 'L-BFGS-B' with finite diff might struggle with 'min'.
    # 'Nelder-Mead' is safer for non-smooth.
    
    print("Starting local refinement (Nelder-Mead)...")
    result_local = minimize(
        objective, 
        best_centers_flat, 
        method='Nelder-Mead', 
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    final_centers = result_local.x.reshape(-1, 2)
    
    # 5. Compute final radii
    final_radii = get_max_radii(final_centers)
    sum_radii = np.sum(final_radii)
    
    print(f"Final sum of radii: {sum_radii}")
    
    return final_centers, final_radii, sum_radii

# To verify locally if needed, but the function must be self-contained.
# If running as script:
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Simple validation check
    # (Note: full validation requires the provided function, but we can do a quick check)
    # Check overlaps
    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    overlaps = (min_dists < 2 * radii - 1e-12)
    if np.any(overlaps):
        print("Warning: Overlaps detected!")
    else:
        print("No overlaps detected.")
