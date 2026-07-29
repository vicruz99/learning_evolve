# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 38145db4) state=45ce8876 sum of radii=0.649967 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_repulsion_forces(centers, radii):
    """Compute repulsive forces between overlapping circles."""
    N = centers.shape[0]
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    min_dists = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Prevent division by zero
    dists_safe = np.where(dists < 1e-9, 1e-9, dists)
    overlap = np.maximum(0, min_dists - dists_safe)
    
    # Direction of repulsion
    force_dir = diffs / dists_safe[:, :, np.newaxis]
    force_mag = overlap * 100.0  # Spring stiffness
    forces = np.sum(force_mag[:, :, np.newaxis] * force_dir, axis=1)
    return forces, np.max(overlap)

def apply_boundary_forces(centers, radii):
    """Compute repulsive forces from the unit square boundaries."""
    N = centers.shape[0]
    forces = np.zeros_like(centers)
    max_ov = 0.0
    
    for i in range(N):
        r = radii[i]
        cx, cy = centers[i]
        
        if cx < r:
            forces[i, 0] += (r - cx) * 200.0
            max_ov = max(max_ov, r - cx)
        if cx > 1 - r:
            forces[i, 0] -= (cx - (1 - r)) * 200.0
            max_ov = max(max_ov, cx - (1 - r))
        if cy < r:
            forces[i, 1] += (r - cy) * 200.0
            max_ov = max(max_ov, r - cy)
        if cy > 1 - r:
            forces[i, 1] -= (cy - (1 - r)) * 200.0
            max_ov = max(max_ov, cy - (1 - r))
            
    return forces, max_ov

def run_packing():
    """Pack 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    N = 26
    
    # Initialize centers in a hexagonal pattern
    centers = np.zeros((N, 2))
    idx = 0
    rows = [5, 4, 5, 4, 5, 3]  # Sum = 26
    dx = 0.18
    dy = 0.18 * np.sqrt(3) / 2
    y = 0.10
    
    for i, count in enumerate(rows):
        # Alternate rows are shifted for hexagonal packing
        offset = (dx / 2.0) if (i % 2 == 1) else 0.0
        x_start = 0.10 + offset
        for j in range(count):
            if idx < N:
                centers[idx, 0] = x_start + j * dx
                centers[idx, 1] = y
                idx += 1
        y += dy
        
    # Start with small radii
    radii = np.full(N, 0.025)
    velocities = np.zeros_like(centers)
    
    # Simulation loop
    for step in range(40000):
        # Compute forces
        rep_forces, ov1 = compute_repulsion_forces(centers, radii)
        bnd_forces, ov2 = apply_boundary_forces(centers, radii)
        
        forces = rep_forces + bnd_forces
        
        # Update dynamics
        velocities = 0.35 * velocities + forces * 0.015
        centers += velocities
        
        # Keep centers within [0,1] strictly
        centers = np.clip(centers, 0.0, 1.0)
        
        # Grow radii when packing is stable
        max_ov = max(ov1, ov2)
        if max_ov < 1e-8:
            radii *= 1.00005
            
    # Final safety shrink to guarantee validation passes within tolerance
    radii *= 0.99995
    
    total_radius = float(np.sum(radii))
    return centers, radii, total_radius
