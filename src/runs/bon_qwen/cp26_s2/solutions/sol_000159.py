# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a0a8497a) state=afd79db5 sum of radii=2.166040 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_radii(centers):
    """
    Compute the maximum feasible radius for each circle given fixed centers.
    Vectorized for performance.
    """
    n = centers.shape[0]
    # Distance to boundaries
    dx = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    dy = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    r_boundary = np.minimum(dx, dy)

    # Distance to other circles
    # Compute pairwise differences: shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    r_neighbor = np.min(dists, axis=1) / 2.0

    # Final radius is limited by both boundaries and neighbors
    return np.minimum(r_boundary, r_neighbor)

def run_simulation(init_centers, steps=3000, r_target_final=0.11):
    """
    Run repulsion-based simulation to pack circles tightly.
    Returns optimized centers.
    """
    centers = init_centers.copy()
    n = centers.shape[0]
    
    # Precompute constants
    step_sizes = 0.05 * (1 - np.arange(steps) / steps)**2
    
    for step in range(steps):
        r_curr = r_target_final * (step + 100) / (steps + 100)
        step_size = step_sizes[step]
        
        # 1. Compute inter-circle forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Direction vectors
        dirs = diff / dists[..., np.newaxis]
        dirs[dirs != dirs] = 0.0  # Handle division by zero
        
        # Overlap amount
        overlap = np.maximum(0.0, 2.0 * r_curr - dists)
        overlap = np.where(dists < 1e-8, 0.0, overlap) # Safety
        
        # Force magnitude proportional to overlap
        forces = np.sum(dirs * overlap[..., np.newaxis], axis=1)
        
        # 2. Compute boundary forces
        bforce = np.zeros_like(centers)
        margin = r_curr
        
        # Left wall
        mask = centers[:, 0] < margin
        bforce[mask, 0] += (margin - centers[mask, 0])
        # Right wall
        mask = centers[:, 0] > 1.0 - margin
        bforce[mask, 0] -= (centers[mask, 0] - (1.0 - margin))
        # Bottom wall
        mask = centers[:, 1] < margin
        bforce[mask, 1] += (margin - centers[mask, 1])
        # Top wall
        mask = centers[:, 1] > 1.0 - margin
        bforce[mask, 1] -= (centers[mask, 1] - (1.0 - margin))
        
        # Combine forces
        total_force = forces + bforce
        
        # Update positions
        centers += total_force * step_size
        
        # Clamp to valid range to prevent numerical drift
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main entry point. Runs multiple optimization trials and returns the best packing.
    """
    np.random.seed(42)
    best_sum = -1.0
    best_centers = None
    
    # Try multiple restarts to avoid local optima
    for _ in range(8):
        # Initialize on a hexagonal grid
        # 5 rows: 5, 6, 5, 6, 4 = 26 circles
        centers = []
        y = 0.12
        row_counts = [5, 6, 5, 6, 4]
        for count in row_counts:
            x_start = 0.12 if count % 2 == 1 else 0.23
            x_end = 1.0 - x_start
            xs = np.linspace(x_start, x_end, count)
            centers.append(np.column_stack([xs, np.full(count, y)]))
            y += 0.18
        
        centers = np.vstack(centers)
        
        # Add small random perturbation
        centers += np.random.randn(26, 2) * 0.015
        centers = np.clip(centers, 0.02, 0.98)
        
        # Optimize
        opt_centers = run_simulation(centers, steps=2500, r_target_final=0.115)
        
        # Compute actual feasible radii
        radii = compute_radii(opt_centers)
        total_sum = np.sum(radii)
        
        if total_sum > best_sum:
            best_sum = total_sum
            best_centers = opt_centers.copy()
            
    # Final radii calculation
    radii = compute_radii(best_centers)
    total_sum = float(np.sum(radii))
    
    return best_centers, radii, total_sum
