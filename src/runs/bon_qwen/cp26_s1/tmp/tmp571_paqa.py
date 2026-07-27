import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)  # For reproducibility
    
    # Initialize positions and radii
    centers = np.random.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.01)
    velocities = np.zeros((n, 2))
    
    # Simulation parameters
    max_iter = 1500
    growth_rate_init = 0.0008
    damping = 0.92
    repulsion_scale = 50.0
    wall_stiffness = 100.0

    for i in range(max_iter):
        # Slowly decrease growth rate to settle
        growth = growth_rate_init * (1.0 + 0.001 * i) ** (-0.6)
        
        # Grow radii
        radii += growth
        
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Vectorized inter-circle forces
        dists = np.sqrt(np.sum((centers[:, np.newaxis] - centers) ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        overlaps = radii[:, np.newaxis] + radii - dists
        overlap_mask = overlaps > 0
        
        # Calculate repulsion forces only for overlapping pairs
        overlap_magnitudes = np.zeros_like(dists)
        np.fill_diagonal(overlap_magnitudes, 0)
        overlap_magnitudes[overlap_mask] = overlaps[overlap_mask] ** 2 * repulsion_scale
        
        # Directions
        dirs = (centers[:, np.newaxis] - centers)
        np.fill_diagonal(dirs, np.inf)
        dirs[overlap_mask] /= dists[overlap_mask]
        
        forces += np.sum(dirs * overlap_magnitudes, axis=1)
        
        # Wall forces
        # Left wall
        violations = radii - centers[:, 0]
        mask = violations > 0
        forces[mask, 0] += violations[mask] * wall_stiffness
        # Right wall
        violations = radii - (1.0 - centers[:, 0])
        mask = violations > 0
        forces[mask, 0] -= violations[mask] * wall_stiffness
        # Bottom wall
        violations = radii - centers[:, 1]
        mask = violations > 0
        forces[mask, 1] += violations[mask] * wall_stiffness
        # Top wall
        violations = radii - (1.0 - centers[:, 1])
        mask = violations > 0
        forces[mask, 1] -= violations[mask] * wall_stiffness

        # Update velocities and positions
        velocities += forces * 0.0001
        velocities *= damping
        centers += velocities
        
        # Clamp centers to keep simulation stable
        centers = np.clip(centers, radii[:, np.newaxis], 1.0 - radii[:, np.newaxis])

    # Post-processing: ensure strict validity and maximize radii based on final centers
    # Re-calculate maximum possible radii given the centers
    for _ in range(10):
        for i in range(n):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    max_r = min(max_r, dist - radii[j])
            radii[i] = max(0, max_r)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii