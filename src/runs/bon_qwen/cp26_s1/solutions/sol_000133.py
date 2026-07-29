# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5da4630c) state=52c4fc2f sum of radii=2.006950 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_smooth_bottleneck(centers_flat, k):
    """
    Computes the negative smooth approximation of the minimum distance.
    We minimize this to maximize the bottleneck distance.
    """
    centers = centers_flat.reshape(26, 2)
    n = 26
    
    # Collect all distance constraints
    dists = []
    
    # Boundary distances: min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        dists.append(x)
        dists.append(1 - x)
        dists.append(y)
        dists.append(1 - y)
        
    # Inter-circle distances
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists.append(d)
            
    dists = np.array(dists)
    # Smooth minimum approximation
    # We use log-sum-exp to approximate -min(d_i)
    # Objective: (1/k) * log(sum(exp(-k * d_i)))
    # Minimizing this pushes all d_i up
    return (1.0 / k) * np.log(np.sum(np.exp(-k * dists)))

def run_packing():
    # 1. Initialization: Hexagonal grid perturbed
    np.random.seed(42)
    centers = np.zeros((26, 2))
    
    # Create a dense 6x5 grid and remove 4 to get 26
    # This provides good initial spacing
    x_vals = np.linspace(0.15, 0.85, 6)
    y_vals = np.linspace(0.15, 0.85, 5)
    
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx >= 26:
                break
            centers[idx, 0] = x_vals[i]
            centers[idx, 1] = y_vals[j]
            idx += 1
        if idx >= 26:
            break
            
    # Add small random noise to break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    
    # 2. Repulsion Dynamics (Jiggle)
    # This phase rapidly resolves overlaps and pushes circles into optimal relative positions
    current_r = 0.08
    step_size = 0.05
    alpha = 0.9995  # Cooling factor for step size
    
    for _ in range(15000):
        forces = np.zeros_like(centers)
        overlap_sum = 0.0
        
        n = 26
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                target_dist = 2.0 * current_r
                if dist < target_dist and dist > 1e-8:
                    # Repulsive force proportional to overlap
                    f_mag = (target_dist - dist) / dist
                    forces[i] += diff * f_mag
                    forces[j] -= diff * f_mag
                    overlap_sum += (target_dist - dist)
                    
            # Boundary forces
            x, y = centers[i]
            if x < current_r:
                forces[i, 0] += (current_r - x) * 10.0
            if x > 1.0 - current_r:
                forces[i, 0] -= (x - (1.0 - current_r)) * 10.0
            if y < current_r:
                forces[i, 1] += (current_r - y) * 10.0
            if y > 1.0 - current_r:
                forces[i, 1] -= (y - (1.0 - current_r)) * 10.0
                
        # Normalize and apply forces
        norms = np.linalg.norm(forces, axis=1)
        norms[norms < 1e-9] = 1.0
        forces_normalized = forces / norms[:, np.newaxis]
        
        centers += step_size * forces_normalized
        centers = np.clip(centers, 0.0, 1.0)
        
        # Adaptive radius control
        if overlap_sum < 1e-5:
            current_r *= 1.0002
        else:
            current_r *= 0.999
            
        step_size *= alpha
        
    # 3. Gradient-based Refinement
    # Use L-BFGS-B to optimize the smooth bottleneck objective
    # This polishes the configuration to a local optimum
    k_val = 80.0  # Smoothing parameter
    bounds = [(0.0, 1.0) for _ in range(52)]
    
    res = minimize(
        fun=compute_smooth_bottleneck,
        x0=centers.flatten(),
        args=(k_val,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9}
    )
    
    optimal_centers = res.x.reshape(26, 2)
    
    # 4. Compute final true bottleneck radius
    min_dist = 1.0
    
    # Check boundaries
    d_x = np.minimum(optimal_centers[:, 0], 1.0 - optimal_centers[:, 0])
    d_y = np.minimum(optimal_centers[:, 1], 1.0 - optimal_centers[:, 1])
    min_dist = min(min_dist, np.min(d_x), np.min(d_y))
    
    # Check inter-circle distances
    for i in range(26):
        for j in range(i + 1, 26):
            d = np.linalg.norm(optimal_centers[i] - optimal_centers[j])
            if d < min_dist:
                min_dist = d
                
    final_radius = min_dist / 2.0
    radii = np.full(26, final_radius)
    sum_radii = 26 * final_radius
    
    return optimal_centers, radii, sum_radii
