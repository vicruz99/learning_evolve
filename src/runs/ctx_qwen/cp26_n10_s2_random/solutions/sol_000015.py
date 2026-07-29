# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 38145db4) state=9892de0a sum of radii=2.426553 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def compute_objective(centers):
    """
    Computes the maximum possible radii for a given set of centers 
    such that circles do not overlap and stay within the unit square.
    Returns the radii array and their sum.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise Euclidean distances
    # Broadcasting: diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distances by setting diagonal to infinity
    np.fill_diagonal(dist, np.inf)
    
    # Minimum distance to any other circle
    min_dist = np.min(dist, axis=1)
    
    # Radius constrained by half the minimum distance to avoid overlap
    r_pair = 0.5 * min_dist
    
    # The actual radius is the limiting factor between boundary and neighbors
    radii = np.minimum(r_bound, r_pair)
    
    return radii, np.sum(radii)

def objective_function(x):
    """
    Objective function for minimizers. 
    Minimizers seek to minimize, so we return the negative sum of radii.
    """
    centers = x.reshape(-1, 2)
    _, sum_radii = compute_objective(centers)
    return -sum_radii

def generate_initial_centers(n):
    """
    Generates an initial configuration of n centers using a hexagonal grid pattern.
    This arrangement is known for high packing density.
    """
    side = math.ceil(math.sqrt(n))
    points = []
    
    # Estimate spacing for hexagonal packing in unit area
    # Area per circle ~ 1/n. For hex grid, area = sqrt(3)/2 * s^2 => s ~ sqrt(2/(sqrt(3)*n))
    s = math.sqrt(2.0 / (math.sqrt(3.0) * n))
    
    row = 0
    while len(points) < n:
        col = 0
        while len(points) < n:
            x = col * s + (0.5 * s if row % 2 == 1 else 0.0)
            y = row * s * math.sqrt(3.0) / 2.0
            points.append([x, y])
            col += 1
        row += 1
        
    pts = np.array(points[:n])
    
    # Normalize to [0.05, 0.95] to keep away from exact boundaries initially
    min_p = pts.min(axis=0)
    max_p = pts.max(axis=0)
    pts = (pts - min_p) / (max_p - min_p + 1e-12)
    pts = pts * 0.9 + 0.05
    
    # Add small random noise to break symmetry and avoid degenerate cases
    pts += np.random.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.0, 1.0)
    
    return pts

def run_packing():
    """
    Main function to pack 26 circles in a unit square maximizing sum of radii.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers
    centers = generate_initial_centers(n)
    current_centers = centers.copy()
    radii, current_sum = compute_objective(current_centers)
    
    # 2. Simulated Annealing Parameters
    num_iterations = 15000
    initial_temp = 0.05
    step_size = 0.03
    local_opt_freq = 500
    
    best_sum = current_sum
    best_centers = current_centers.copy()
    
    for i in range(num_iterations):
        # Cooling schedule
        temp = initial_temp * (1.0 - i / num_iterations) ** 2
        if temp < 1e-7:
            temp = 1e-7
            
        # Perturb a random circle
        idx = np.random.randint(n)
        new_centers = current_centers.copy()
        perturbation = np.random.uniform(-step_size, step_size, 2)
        new_centers[idx] += perturbation
        
        # Ensure centers stay within [0,1]
        new_centers[idx] = np.clip(new_centers[idx], 0.0, 1.0)
        
        _, new_sum = compute_objective(new_centers)
        delta = new_sum - current_sum
        
        # Acceptance criterion
        if delta > 0 or np.random.rand() < math.exp(delta / temp):
            current_centers = new_centers
            current_sum = new_sum
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = current_centers.copy()
                
                # Periodic local optimization to refine the best found configuration
                if i % local_opt_freq == 0:
                    x_flat = best_centers.flatten()
                    try:
                        # Powell method is robust for non-smooth functions
                        res = minimize(objective_function, x_flat, method='Powell', 
                                       options={'maxiter': 500, 'ftol': 1e-9})
                        if -res.fun > best_sum:
                            opt_centers = res.x.reshape(-1, 2)
                            _, opt_sum = compute_objective(opt_centers)
                            if opt_sum > best_sum:
                                best_sum = opt_sum
                                best_centers = opt_centers
                                current_centers = best_centers.copy()
                                current_sum = best_sum
                    except Exception:
                        pass 

    # Final refinement
    x_flat = best_centers.flatten()
    try:
        res = minimize(objective_function, x_flat, method='Powell', 
                       options={'maxiter': 2000, 'ftol': 1e-10})
        if -res.fun > best_sum:
            best_centers = res.x.reshape(-1, 2)
            _, best_sum = compute_objective(best_centers)
    except Exception:
        pass
        
    # Compute final radii for the optimal centers
    radii, final_sum = compute_objective(best_centers)
    
    return best_centers, radii, float(final_sum)
