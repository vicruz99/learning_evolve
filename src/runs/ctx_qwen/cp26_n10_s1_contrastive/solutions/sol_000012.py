# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 866072f0) state=b688e397 sum of radii=1.279118 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution, minimize

def calculate_max_radius(centers):
    """
    Calculates the maximum possible equal radius for a given set of centers
    such that circles do not overlap and stay within the unit square.
    """
    # Boundary distances: distance to the closest wall
    # x, 1-x, y, 1-y
    dists_x = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    dists_y = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    boundary_dists = np.minimum(dists_x, dists_y)
    min_boundary_dist = np.min(boundary_dists)
    
    # Pairwise distances
    # Compute all pairwise distances
    # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    dists[np.eye(dists.shape[0], dtype=bool)] = np.inf
    
    # Minimum distance between centers
    min_center_dist = np.min(dists)
    
    # The radius is limited by half the center distance and the boundary distance
    radius = min(min_center_dist / 2.0, min_boundary_dist)
    
    return radius

def objective(params):
    """
    Objective function for optimization.
    Minimizes the negative of the max radius (equivalent to maximizing radius).
    """
    centers = params.reshape(26, 2)
    # Ensure coordinates are within bounds for calculation, 
    # though optimizer bounds should handle this.
    # Clamping just in case of numerical issues, though bounds are strict.
    centers = np.clip(centers, 0, 1)
    
    r = calculate_max_radius(centers)
    return -r # We want to maximize r, so minimize -r

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions of 26 circles in a unit square to maximize the sum of radii.
    Assumes circles will be of equal radius for optimal density.
    """
    n_circles = 26
    
    # Define bounds for each coordinate (x, y) in [0, 1]
    bounds = [(0, 1) for _ in range(2 * n_circles)]
    
    # Use Differential Evolution for global optimization
    # population size scaling factor: 15 * 52 is a lot, but default is 15*ndim? 
    # Default popsize is 15 * ndim. For 52 vars, that's 780 individuals. Might be slow.
    # Let's reduce popsize multiplier or use shgo/basinhopping?
    # DE is robust. Let's try with reasonable settings.
    
    # To save time, we can use a smaller population or fewer iterations, 
    # but we need a good solution.
    # Let's try a hybrid: Random restarts with local optimization might be faster for this dimension.
    
    best_params = None
    best_score = np.inf # Minimizing -radius
    
    # Strategy 1: Differential Evolution with limited iterations
    # It might be too slow for 52 vars in a restricted environment.
    # Let's try Basin-hopping or just multiple Nelder-Mead runs from good starts.
    
    # Strategy 2: Multiple restarts of Nelder-Mead from grid/heuristic starts
    # This is often faster for packing problems.
    
    np.random.seed(42)
    
    for _ in range(10): # 10 restarts
        # Initialize centers on a perturbed grid
        # A 6x5 grid has 30 points. We need 26.
        # We can place points on a hexagonal lattice or just a dense grid.
        
        # Generate a grid
        x_coords = np.linspace(0.05, 0.95, 6)
        y_coords = np.linspace(0.05, 0.95, 5)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        
        # Shuffle and pick 26
        np.random.shuffle(grid_points)
        init_centers = grid_points[:26].copy()
        
        # Add some random noise to escape local minima of the grid
        init_centers += np.random.uniform(-0.05, 0.05, size=init_centers.shape)
        init_centers = np.clip(init_centers, 0, 1)
        
        # Optimize using Nelder-Mead (derivative-free)
        # It handles the non-smooth min function reasonably well.
        res = minimize(objective, init_centers.flatten(), method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    # Extract best centers
    centers_opt = best_params.reshape(26, 2)
    
    # Calculate the final radius based on the optimized centers
    # We must recalculate because the optimizer minimizes -radius, 
    # but numerical precision might leave slight overlaps.
    # However, the objective function computes the valid radius.
    # We should clamp centers to ensure validity and then compute radius.
    
    centers_final = np.clip(centers_opt, 0, 1)
    final_radius = calculate_max_radius(centers_final)
    
    # If the calculated radius is very close to the objective value (negated), good.
    # If overlaps exist due to numerical error in optimizer (unlikely with clipping),
    # the calculate_max_radius handles it by finding the limiting distance.
    
    radii = np.full(26, final_radius)
    sum_radii = 26 * final_radius
    
    return centers_final, radii, sum_radii

# Validation helper (read-only as per prompt, but useful for local testing if I were running it)
# def validate_packing(centers, radii): ...

if __name__ == "__main__":
    # Test run
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Radius: {radii[0]}")
    
    # Quick validation logic inline to ensure no errors
    n = centers.shape[0]
    valid = True
    # Check NaN
    if np.isnan(centers).any() or np.isnan(radii).any():
        valid = False
    # Check negative radii
    if (radii < 0).any():
        valid = False
    # Check bounds
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                break
    print(f"Valid: {valid}")
