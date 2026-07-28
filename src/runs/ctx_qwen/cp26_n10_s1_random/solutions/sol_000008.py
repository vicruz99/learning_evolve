# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state abc5794a) state=039b85e0 sum of radii=1.328137 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)

    # --- Step 1: Generate Initial Hexagonal Lattice ---
    # We generate a larger lattice and pick the best candidates to start with.
    density = 0.18  # Initial spacing
    rows = int(np.ceil(1.0 / (density * np.sqrt(3)/2)) + 2)
    cols = int(np.ceil(1.0 / density) + 2)
    
    points = []
    for r in range(rows):
        y = r * density * np.sqrt(3)/2
        for c in range(cols):
            x = c * density + (r % 2) * (density / 2)
            points.append([x, y])
    
    initial_points = np.array(points)
    
    # Center the lattice in the unit square
    cx, cy = np.mean(initial_points, axis=0)
    initial_points -= np.array([cx - 0.5, cy - 0.5])
    
    # Clip to ensure they are roughly in [0, 1] before optimization
    initial_points = np.clip(initial_points, 0.05, 0.95)
    
    # Sort by distance from center to pick the best N points
    dists = np.sum((initial_points - 0.5)**2, axis=1)
    indices = np.argsort(dists)
    best_indices = indices[:n]
    centers = initial_points[best_indices]

    # --- Step 2: Optimize Centers to Maximize Minimum Distance ---
    # We maximize the minimum of (pairwise distance, distance to boundary)
    
    def min_dist(centers_flat):
        centers_2d = centers_flat.reshape(-1, 2)
        # Pairwise distances
        d = np.sqrt(np.sum((centers_2d[:, np.newaxis] - centers_2d[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pairwise = np.min(d)
        
        # Distance to boundaries
        dists_to_wall = np.minimum(
            np.minimum(centers_2d[:, 0], 1 - centers_2d[:, 0]),
            np.minimum(centers_2d[:, 1], 1 - centers_2d[:, 1])
        )
        min_boundary = np.min(dists_to_wall)
        
        return min(min_pairwise, min_boundary)

    # Optimization
    res = scipy.optimize.minimize(
        lambda x: -min_dist(x),
        x0=centers.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12}
    )
    
    optimized_centers = res.x.reshape(-1, 2)

    # --- Step 3: Calculate Final Radii ---
    # The radius of each circle is limited by its closest neighbor and its boundaries.
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        min_wall = min(optimized_centers[i, 0], 1 - optimized_centers[i, 0],
                       optimized_centers[i, 1], 1 - optimized_centers[i, 1])
        # Distance to other centers (pairwise)
        dists = np.sqrt(np.sum((optimized_centers - optimized_centers[i])**2, axis=1))
        dists[i] = np.inf
        min_pair = np.min(dists) / 2.0
        
        # Conservative radius assignment to ensure no overlap
        radii[i] = min(min_wall, min_pair)

    # Add a tiny buffer to prevent floating point errors from triggering validation failure
    radii *= 0.99999
    centers = np.clip(optimized_centers, 1e-9, 1 - 1e-9)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
