# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ad74c980) state=f1cedcb6 sum of radii=0.904528 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def get_max_radii(centers):
    """
    Computes the maximum valid radius for each circle given fixed centers.
    r_i = min(distance to walls, 0.5 * distance to nearest neighbor)
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Precompute distances to walls
    # x in [0, 1], y in [0, 1]
    # dist to left: x, right: 1-x, bottom: y, top: 1-y
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Distance to nearest neighbor
    # Compute pairwise distance matrix
    # centers is (n, 2)
    # diff is (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    # Nearest neighbor distance for each i
    nn_dists = np.min(dists, axis=1)
    
    # Max radius is limited by walls and half the nearest neighbor distance
    radii = np.minimum(wall_dists, 0.5 * nn_dists)
    
    return radii

def objective(centers_flat):
    """
    Objective function to maximize sum of radii.
    Returns negative sum for minimization.
    """
    centers = centers_flat.reshape(-1, 2)
    radii = get_max_radii(centers)
    return -np.sum(radii)

def force_directed_layout(n, iterations=500):
    """
    Initializes n points and runs a repulsion simulation to spread them out.
    """
    # Initialize on a grid or random
    # A dense grid might be better to start
    rng = np.random.default_rng(42)
    centers = rng.random((n, 2))
    
    # Repulsion strength
    k = 10.0
    dt = 0.01
    damping = 0.9
    
    # Run simulation
    for _ in range(iterations):
        velocities = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-5:
                    dist = 1e-5 # Avoid singularity
                
                # Repulsive force
                force = k / (dist**2)
                direction = diff / dist
                force_vec = force * direction
                
                velocities[i] += force_vec
                velocities[j] -= force_vec
            
            # Wall repulsion (keep inside [0,1])
            for d in range(2):
                if centers[i, d] < 0.05:
                    velocities[i, d] += k * (0.05 - centers[i, d])
                elif centers[i, d] > 0.95:
                    velocities[i, d] -= k * (centers[i, d] - 0.95)
        
        # Update positions
        centers += velocities * dt
        centers = np.clip(centers, 0.0, 1.0)
        # Apply damping to velocities? 
        # Actually we recalculate velocities each step based on positions, 
        # so simple Euler integration is fine. 
        # But to stabilize, we can reduce step size or add damping to displacement.
        
    return centers

def run_packing():
    # Number of circles
    N = 26
    
    # 1. Generate a good initial configuration using force-directed layout
    # We run a few iterations to spread points
    initial_centers = force_directed_layout(N, iterations=300)
    
    # 2. Optimize centers to maximize sum of radii
    # Flatten centers for scipy
    x0 = initial_centers.flatten()
    
    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Use Nelder-Mead as it doesn't require gradients and handles non-smooth objectives
    # The objective function is -sum(radii), radii computed from centers.
    # It is non-smooth due to 'min' operations, but usually works okay.
    
    # To improve robustness, we can try a few random restarts or just rely on the good init
    # Let's try optimizing from the force-directed start
    result = minimize(
        objective, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    optimized_centers = result.x.reshape(-1, 2)
    optimized_radii = get_max_radii(optimized_centers)
    
    # Check if we can improve by trying a slightly different initial grid
    # Hexagonal grid init might be better for packing
    # Let's try a hexagonal lattice initialization as well and compare
    
    # Generate hexagonal lattice points
    hex_points = []
    r_approx = 0.1 # Guess
    # We need to fit 26 points. Let's generate a dense set and pick 26.
    # Or just place them in rows.
    # 5 rows of roughly 5-6 points.
    
    # Let's try a specific structured init:
    # Rows y = 0.1, 0.3, 0.5, 0.7, 0.9
    # Points x spaced by ~0.2
    hex_centers = []
    y_coords = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85] # 6 rows?
    # Let's just use the force directed result as it's quite robust.
    # But let's verify the sum.
    
    current_sum = np.sum(optimized_radii)
    best_sum = current_sum
    best_centers = optimized_centers
    best_radii = optimized_radii
    
    # Try one more run with a different random seed for force layout
    rng2 = np.random.default_rng(123)
    alt_centers_init = rng2.random((N, 2))
    alt_centers = force_directed_layout(N, iterations=300) # Reuse function but it uses fixed seed inside? 
    # Wait, force_directed_layout uses fixed seed 42. Let's modify to accept rng or just run again.
    # Actually, the function uses np.random.default_rng(42) locally.
    # To vary, I'd need to pass seed. But let's assume the first one is good enough.
    # Or I can just perturb the optimized result and re-optimize.
    
    # Perturbation
    perturbed_centers = best_centers + np.random.normal(0, 0.01, best_centers.shape)
    perturbed_centers = np.clip(perturbed_centers, 0, 1)
    
    result2 = minimize(
        objective,
        perturbed_centers.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7}
    )
    
    c2 = result2.x.reshape(-1, 2)
    r2 = get_max_radii(c2)
    sum2 = np.sum(r2)
    
    if sum2 > best_sum:
        best_sum = sum2
        best_centers = c2
        best_radii = r2
        
    # Final check and return
    # Ensure centers are valid
    best_centers = np.clip(best_centers, 0.0, 1.0)
    # Recalculate radii to be safe (in case of numerical noise moving centers out slightly)
    best_radii = get_max_radii(best_centers)
    
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum

# Helper to ensure the function is top-level and runnable
def solve():
    centers, radii, s = run_packing()
    return centers, radii, s

# The problem asks for run_packing to be defined.
# I will define it as above.
