# sol_000247 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 240eaf0d) state=d7a0dcc7 sum of radii=1.028748 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def get_max_radius(centers):
    """
    Computes the maximum possible radii for circles at given centers
    such that they fit in the unit square and do not overlap.
    """
    n = centers.shape[0]
    # Max radius allowed by boundaries for each circle
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    boundaries = np.array([min(c[0], 1-c[0], c[1], 1-c[1]) for c in centers])
    radii = boundaries.copy()
    
    # Distance matrix between centers
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Iteratively reduce radii to satisfy pairwise non-overlap constraints
    # Constraint: radii[i] + radii[j] <= dists[i, j]
    # We start with maximal radii (boundaries) and reduce only when necessary.
    # To maximize sum of radii, we distribute reductions evenly between pairs.
    
    for _ in range(100):
        changed = False
        # Compute current violations: (r_i + r_j) - dist_ij
        # We only care if positive
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        violations = r_sum - dists
        
        # Find pairs that are overlapping (violation > tolerance)
        threshold = 1e-12
        idx = np.where(violations > threshold)
        
        if len(idx[0]) == 0:
            break
            
        # Process each violating pair
        # Note: Modifying radii in loop might affect other pairs, 
        # so multiple iterations are needed.
        for i, j in zip(idx[0], idx[1]):
            if i < j:
                # Recalculate excess to be precise
                excess = (radii[i] + radii[j] - dists[i, j])
                if excess > threshold:
                    # Reduce both radii by half the excess to maintain balance
                    reduction = excess / 2.0
                    radii[i] -= reduction
                    radii[j] -= reduction
                    changed = True
        
        if not changed:
            break
            
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    return radii


def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialization
    # Create a hexagonal lattice arrangement for better packing density
    # Generate a grid of points
    points = []
    # Using a spacing that allows roughly 5x5 points
    spacing = 0.15 
    for i in range(8):
        for j in range(8):
            x = i * spacing + (spacing * 0.5 if j % 2 == 1 else 0.0)
            y = j * spacing * math.sqrt(3) / 2
            points.append([x, y])
    
    points = np.array(points)
    
    # Select 26 points closest to the center to form a compact cluster
    center_pt = np.mean(points, axis=0)
    dists_from_center = np.linalg.norm(points - center_pt, axis=1)
    indices = np.argsort(dists_from_center)[:n]
    selected = points[indices]
    
    # Scale and translate the cluster to fit within the unit square with margins
    min_c = np.min(selected, axis=0)
    max_c = np.max(selected, axis=0)
    size = max_c - min_c
    
    # Margin to allow circles to have radius
    margin = 0.15
    # Scale factor to fit into [margin, 1-margin]
    scale = (1 - 2 * margin) / np.max(size)
    
    centers = np.zeros((n, 2))
    centers[:, 0] = (selected[:, 0] - min_c[0]) * scale + margin
    centers[:, 1] = (selected[:, 1] - min_c[1]) * scale + margin
    
    radii = np.full(n, 0.01)
    
    # 2. Simulation / Optimization
    # We simulate repulsive forces to spread circles and grow radii
    steps = 5000
    dt = 1e-5
    repulsion_k = 100.0
    boundary_k = 200.0
    growth_rate = 2e-5 
    
    for step in range(steps):
        # Gradually increase radii to push circles apart
        radii += growth_rate
        
        # Compute pairwise repulsion forces
        # diff[i, j] = center_i - center_j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-9) # Avoid division by zero
        
        # Overlap amount
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sum - dists
        overlap = np.maximum(overlap, 0)
        
        # Direction of repulsion (unit vector)
        safe_dist = dists[:, :, np.newaxis]
        directions = diff / safe_dist
        
        # Force magnitude proportional to overlap
        force_mag = overlap * repulsion_k
        force_vec = directions * force_mag[:, :, np.newaxis]
        
        # Sum forces acting on each circle
        forces = np.sum(force_vec, axis=1)
        
        # Boundary repulsion forces
        # If circle overlaps with wall, push it back inside
        # Left wall (x < r)
        mask = centers[:, 0] < radii
        forces[mask, 0] += (radii[mask] - centers[mask, 0]) * boundary_k
        # Right wall (x > 1-r)
        mask = centers[:, 0] > 1 - radii
        forces[mask, 0] -= (centers[mask, 0] - (1 - radii[mask])) * boundary_k
        # Bottom wall (y < r)
        mask = centers[:, 1] < radii
        forces[mask, 1] += (radii[mask] - centers[mask, 1]) * boundary_k
        # Top wall (y > 1-r)
        mask = centers[:, 1] > 1 - radii
        forces[mask, 1] -= (centers[mask, 1] - (1 - radii[mask])) * boundary_k
        
        # Update positions
        centers += forces * dt
        
        # Clip positions to stay within [0, 1] (safety measure)
        centers[:, 0] = np.clip(centers[:, 0], 1e-7, 1-1e-7)
        centers[:, 1] = np.clip(centers[:, 1], 1e-7, 1-1e-7)
        
        # Reduce growth rate slightly over time to allow settling
        if step % 1000 == 0 and step > 0:
            growth_rate *= 0.8
            
    # 3. Final adjustment of radii
    # Compute the maximal valid radii for the optimized centers
    radii = get_max_radius(centers)
    
    return centers, radii, np.sum(radii)
