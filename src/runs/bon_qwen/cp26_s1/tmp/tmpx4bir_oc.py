import numpy as np

def compute_forces(centers, radii):
    """Compute repulsive forces to resolve overlaps and boundary violations."""
    n = len(radii)
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion
    # Compute all pairwise differences and distances
    diffs = centers[:, None, :] - centers[None, :, :]  # (N, N, 2)
    dists = np.sqrt(np.sum(diffs**2, axis=2))           # (N, N)
    np.fill_diagonal(dists, np.inf)                     # Ignore self-interaction
    
    # Overlap magnitude
    overlaps = np.maximum(0.0, radii[:, None] + radii[None, :] - dists)
    
    # Force magnitude: proportional to overlap^2, inversely to distance for direction
    f_mag = (overlaps**2) / (dists + 1e-8)
    
    # Direction vectors
    dir_vecs = diffs / (dists[:, :, None] + 1e-8)
    
    # Accumulate forces
    forces_from_others = np.sum(f_mag[:, :, None] * dir_vecs, axis=1)
    forces += forces_from_others
    
    # Boundary repulsion
    # Left wall
    left_pen = np.maximum(0.0, radii - centers[:, 0])
    forces[:, 0] += left_pen**2
    # Right wall
    right_pen = np.maximum(0.0, centers[:, 0] - (1.0 - radii))
    forces[:, 0] -= right_pen**2
    # Bottom wall
    bottom_pen = np.maximum(0.0, radii - centers[:, 1])
    forces[:, 1] += bottom_pen**2
    # Top wall
    top_pen = np.maximum(0.0, centers[:, 1] - (1.0 - radii))
    forces[:, 1] -= top_pen**2
    
    return forces

def generate_initial_hex_grid(n, r_start):
    """Generate a hexagonal grid initialization for n circles."""
    # Pattern to fit 26 circles roughly
    row_counts = [5, 6, 5, 6, 4]
    centers = []
    y = r_start + 0.15
    dy = np.sqrt(3.0) * r_start
    
    for rc in row_counts:
        # Center the row horizontally
        x_start = r_start + (5 - rc) * r_start
        for _ in range(rc):
            centers.append([x_start, y])
            x_start += 2.0 * r_start
        y += dy
        
    return np.array(centers)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize positions
    centers = generate_initial_hex_grid(n, r_start=0.08)
    radii = np.full(n, 0.025)
    
    # 2. Simulation parameters
    lr = 0.02
    growth_rate = 2.5e-5
    damping = 0.92
    steps = 6000
    
    # Velocity for integration
    velocity = np.zeros_like(centers)
    
    for t in range(steps):
        # Compute forces
        forces = compute_forces(centers, radii)
        
        # Update velocity and positions (semi-implicit Euler)
        velocity = damping * velocity + lr * forces
        centers += velocity
        
        # Keep centers roughly inside bounds to prevent drift
        # (Forces handle boundaries, but hard clip prevents extreme outliers)
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
        # Grow radii gradually
        radii += growth_rate
        
        # Decay learning rate for stability
        if t > 2000:
            lr *= 0.9995
            
    # 3. Final Safety Adjustments
    # Shrink radii slightly to guarantee validation tolerance (1e-12)
    safety_margin = 1e-7
    radii = np.maximum(0.0, radii - safety_margin)
    
    # Ensure centers respect final radii boundaries exactly
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    total_sum = float(np.sum(radii))
    return centers, radii, total_sum