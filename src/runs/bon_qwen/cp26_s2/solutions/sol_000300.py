# sol_000300 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d060b5cc) state=255a7109 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed simulation initialized with a perturbed grid pattern.
    """
    n = 26
    # Optimization parameters
    num_steps = 5000
    lr = 0.02          # Initial learning rate for position updates
    current_r = 0.05   # Starting radius
    r_step = 0.00006   # Increment for radius per step
    
    # 1. Initialization
    # Start with a 5x5 grid (25 circles) and place the 26th in a gap.
    # Grid points: 0.1, 0.3, 0.5, 0.7, 0.9
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    
    # Place 26th circle in a gap, e.g., (0.2, 0.2) which is a hole center
    centers.append([0.2, 0.2])
    
    centers = np.array(centers, dtype=float)
    
    # Add small random noise to break symmetry and avoid grid locks
    centers += np.random.randn(n, 2) * 0.005
    centers = np.clip(centers, 0, 1)
    
    # Precompute pair indices for vectorized distance calculation
    i_indices, j_indices = np.triu_indices(n, k=1)
    
    # 2. Simulation Loop
    for step in range(num_steps):
        forces = np.zeros((n, 2))
        
        # Calculate pairwise distances and repulsive forces
        diff = centers[i_indices] - centers[j_indices]
        dists = np.linalg.norm(diff, axis=1)
        
        # Minimum distance required is sum of radii (2*current_r)
        min_dist = 2 * current_r
        
        # Overlap amount: positive if circles are too close
        overlap = min_dist - dists
        overlap = np.maximum(overlap, 0)
        
        # Normalize direction vector
        safe_dists = np.maximum(dists, 1e-7)
        normals = diff / safe_dists[:, np.newaxis]
        
        # Force magnitude proportional to overlap
        force_mag = overlap * 100.0
        forces_vec = force_mag[:, np.newaxis] * normals
        
        # Apply forces to circles
        np.add.at(forces, i_indices, forces_vec)
        np.add.at(forces, j_indices, -forces_vec)
        
        # Boundary forces vectorized
        # Push away from walls if center - r < 0 or center + r > 1
        
        # Left wall (x < r)
        mask = centers[:, 0] < current_r
        forces[mask, 0] += (current_r - centers[mask, 0]) * 100.0
        
        # Right wall (x > 1 - r)
        mask = centers[:, 0] > 1 - current_r
        forces[mask, 0] -= (centers[mask, 0] - (1 - current_r)) * 100.0
        
        # Bottom wall (y < r)
        mask = centers[:, 1] < current_r
        forces[mask, 1] += (current_r - centers[mask, 1]) * 100.0
        
        # Top wall (y > 1 - r)
        mask = centers[:, 1] > 1 - current_r
        forces[mask, 1] -= (centers[mask, 1] - (1 - current_r)) * 100.0
        
        # Update positions
        centers += forces * lr
        centers = np.clip(centers, 0, 1)
        
        # Increase radius slowly
        current_r += r_step
        
        # Cool down learning rate
        lr *= 0.9995
        
    # 3. Final Validation and Adjustment
    # Check for overlaps and boundary violations to ensure strict validity
    
    # Check pairwise overlaps
    diff = centers[i_indices] - centers[j_indices]
    dists = np.linalg.norm(diff, axis=1)
    overlap_violation = np.max(2 * current_r - dists)
    if overlap_violation < 0:
        overlap_violation = 0.0
        
    # Check boundary violations
    # For each circle, r must be <= min(x, 1-x, y, 1-y)
    limits = np.minimum(centers, 1 - centers)
    max_r_boundary = np.min(limits)
    boundary_violation = current_r - max_r_boundary
    if boundary_violation < 0:
        boundary_violation = 0.0
        
    # Determine safety margin to reduce radius
    # For overlap: delta >= overlap_violation / 2
    # For boundary: delta >= boundary_violation
    delta = max(overlap_violation / 2.0, boundary_violation)
    
    if delta > 0:
        delta += 1e-5  # Small buffer for numerical stability
        current_r -= delta
        current_r = max(current_r, 0.0)
        
    radii = np.full(n, current_r)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
