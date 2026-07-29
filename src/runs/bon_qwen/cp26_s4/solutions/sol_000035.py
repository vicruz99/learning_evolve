# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e09efbf) state=ce693b68 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_repulsion_forces(centers):
    """
    Computes repulsive forces between points and from boundaries.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # 1. Pairwise repulsion
    # Calculate pairwise differences: shape (n, n, 2)
    # centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared distances: shape (n, n)
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    # Force magnitude ~ 1/dist^2, direction along diff
    # F = diff / dist^3
    # inv_dist_cubed = 1 / (dist * dist_sq)
    # To prevent explosion when very close, we can cap the force or use soft potential
    # But for packing, hard repulsion is good.
    
    inv_dist_cubed = 1.0 / (dist * dist_sq)
    # inv_dist_cubed shape (n, n)
    # diff shape (n, n, 2)
    # Multiply and sum over j (axis 1)
    # forces += sum_j ( (p_i - p_j) / ||p_i - p_j||^3 )
    forces += np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
    
    # 2. Boundary repulsion
    # Points should stay away from x=0, x=1, y=0, y=1
    # Force pushes away from walls.
    # F_x = 1/x^2 - 1/(1-x)^2
    # F_y = 1/y^2 - 1/(1-y)^2
    # This creates a potential well in the center, but combined with point repulsion,
    # it prevents points from sticking to walls where radius would be 0.
    
    # Clip to avoid singularity if point is exactly on boundary (though simulation keeps them inside)
    eps = 1e-4
    x = np.clip(centers[:, 0], eps, 1 - eps)
    y = np.clip(centers[:, 1], eps, 1 - eps)
    
    fx = (1.0 / (x**2)) - (1.0 / ((1 - x)**2))
    fy = (1.0 / (y**2)) - (1.0 / ((1 - y)**2))
    
    forces[:, 0] += fx
    forces[:, 1] += fy
    
    return forces

def optimize_positions(n_points, steps=3000, seed=42):
    """
    Runs a force-directed simulation to spread points out.
    """
    np.random.seed(seed)
    
    # Initialize positions randomly, slightly inset from boundaries
    # This helps avoid immediate boundary sticking
    centers = np.random.rand(n_points, 2) * 0.8 + 0.1
    
    # Learning rate (step size)
    lr = 0.005
    
    for step in range(steps):
        forces = compute_repulsion_forces(centers)
        
        # Apply forces
        centers += lr * forces
        
        # Project back to unit square [0, 1]
        # Using clip ensures they don't leave, but forces push them inside
        centers = np.clip(centers, 0.0, 1.0)
        
        # Cooling schedule to settle
        lr *= 0.9995
        
    return centers

def calculate_min_separation(centers):
    """
    Calculates the minimum distance between any pair of centers,
    and the minimum distance of any center to the boundary.
    """
    n = centers.shape[0]
    
    # Pairwise distances
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_pair_dist = np.sqrt(min_dist_sq)
    
    # Distance to boundary
    # For each point, dist to boundary is min(x, 1-x, y, 1-y)
    x = centers[:, 0]
    y = centers[:, 1]
    dist_x = np.minimum(x, 1.0 - x)
    dist_y = np.minimum(y, 1.0 - y)
    dist_bound = np.minimum(dist_x, dist_y)
    min_bound_dist = np.min(dist_bound)
    
    return min_pair_dist, min_bound_dist

def run_packing():
    # Number of circles
    n = 26
    
    # Run optimization
    # We can try a couple of seeds or just one robust run.
    # A single run with enough steps usually finds a good lattice-like structure.
    centers = optimize_positions(n, steps=4000, seed=123)
    
    # Calculate limiting distances
    min_pair_dist, min_bound_dist = calculate_min_separation(centers)
    
    # The radius is limited by both pair separation and boundary distance.
    # Circle i and j: dist >= r_i + r_j. If equal radii r, dist >= 2r => r <= dist/2.
    # Circle i boundary: dist_bound >= r => r <= dist_bound.
    r = min(min_pair_dist / 2.0, min_bound_dist)
    
    # Create radii array
    radii = np.full(n, r)
    
    # Sum of radii
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
