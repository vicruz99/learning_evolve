import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Grid
    centers = np.zeros((n_circles, 2))
    # Place in a dense hexagonal arrangement
    count = 0
    y = 0.0
    row = 0
    while count < n_circles:
        x = 0.0
        while count < n_circles:
            # Offset every other row for hexagonal packing
            offset = 0.05 if row % 2 == 1 else 0.0
            centers[count, 0] = x + offset
            centers[count, 1] = y
            count += 1
            x += 0.15
        y += 0.13
        row += 1
        
    # Initial radii
    radii = np.full(n_circles, 0.01)

    # 2. Iterative Expansion and Optimization
    # We will use a loop to grow radii and resolve overlaps using forces
    
    # Physics parameters
    dt = 0.05
    damping = 0.95
    repulsion_strength = 10.0
    boundary_strength = 20.0
    
    # Expansion loop
    for expansion_step in range(1500):
        # Try to increase radii slightly
        # As we progress, increase step size, then decrease
        if expansion_step < 500:
            growth = 0.0001
        else:
            growth = 0.00005
            
        # Attempt to grow all radii
        # To allow for unequal radii optimization later, we could randomize this,
        # but uniform growth finds the equal-radius limit well first.
        radii += growth
        
        # Apply forces to resolve overlaps
        velocities = np.zeros_like(centers)
        
        for _ in range(50): # Internal relaxation steps
            forces = np.zeros_like(centers)
            
            # Circle-Circle repulsion
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    diff = centers[j] - centers[i]
                    dist = np.linalg.norm(diff)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-9:
                        # Overlap detected
                        overlap = min_dist - dist
                        # Force proportional to overlap, inversely proportional to distance
                        force_vec = (diff / dist) * overlap * repulsion_strength
                        forces[i] -= force_vec
                        forces[j] += force_vec
                    elif dist < 1e-9:
                        # Coincident centers, push randomly
                        forces[i] += np.random.randn(2) * 0.01
                        forces[j] -= np.random.randn(2) * 0.01
                        
            # Boundary repulsion
            for i in range(n_circles):
                r = radii[i]
                c = centers[i]
                
                # Left
                if c[0] < r:
                    forces[i, 0] += boundary_strength * (r - c[0])
                # Right
                if c[0] > 1 - r:
                    forces[i, 0] -= boundary_strength * (c[0] - (1 - r))
                # Bottom
                if c[1] < r:
                    forces[i, 1] += boundary_strength * (r - c[1])
                # Top
                if c[1] > 1 - r:
                    forces[i, 1] -= boundary_strength * (c[1] - (1 - r))
            
            # Update velocities and positions
            velocities = velocities * damping + forces * dt
            centers = centers + velocities * dt
            
            # Clamp centers to stay within bounds strictly during simulation
            centers = np.clip(centers, 1e-6, 1 - 1e-6)

        # Random perturbation to escape local minima (Simulated Annealing-like)
        # Perturb radii slightly to allow unequal radii optimization
        if expansion_step % 100 == 0:
            # Perturb radii randomly within a small range
            perturbation = np.random.uniform(-0.001, 0.001, n_circles)
            radii = np.maximum(radii + perturbation, 0.001)
            
            # Also perturb centers slightly
            centers += np.random.randn(n_circles, 2) * 0.001
            centers = np.clip(centers, 0.01, 0.99)

    # 3. Final Validation and Adjustment
    # Ensure no NaNs and valid constraints
    # If any overlaps remain, shrink radii slightly
    max_iter = 100
    for _ in range(max_iter):
        valid = True
        max_overlap = 0
        
        # Check overlaps
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                req_dist = radii[i] + radii[j]
                if dist < req_dist:
                    overlap = req_dist - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                    valid = False
        
        # Check boundaries
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                overlap = max(r - x, x - (1-r), r - y, y - (1-r)) # Simplified
                if overlap > max_overlap:
                    max_overlap = overlap
                valid = False
        
        if valid:
            break
        
        # If invalid, shrink radii by a small fraction of the max overlap
        shrink_factor = max_overlap * 1.1
        radii -= shrink_factor / 2 # Shrink all slightly to resolve
        
        # Re-run a quick force relaxation
        velocities = np.zeros_like(centers)
        for _ in range(20):
            forces = np.zeros_like(centers)
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    diff = centers[j] - centers[i]
                    dist = np.linalg.norm(diff)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist and dist > 1e-9:
                        overlap = min_dist - dist
                        force_vec = (diff / dist) * overlap * repulsion_strength
                        forces[i] -= force_vec
                        forces[j] += force_vec
            
            for i in range(n_circles):
                r = radii[i]
                c = centers[i]
                if c[0] < r: forces[i, 0] += boundary_strength * (r - c[0])
                if c[0] > 1 - r: forces[i, 0] -= boundary_strength * (c[0] - (1 - r))
                if c[1] < r: forces[i, 1] += boundary_strength * (r - c[1])
                if c[1] > 1 - r: forces[i, 1] -= boundary_strength * (c[1] - (1 - r))
            
            velocities = velocities * damping + forces * dt
            centers = centers + velocities * dt
            centers = np.clip(centers, 1e-6, 1 - 1e-6)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii