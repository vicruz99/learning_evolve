import numpy as np
import cvxpy as cp
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a hybrid approach: Physics-based relaxation followed by CVXPY optimization.
    """
    n = 26
    # Constants
    repulsion_k = 500.0
    growth_rate = 0.0002
    damping = 0.85
    
    # 1. Initialize centers in a dense hexagonal-like grid
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.02
    
    idx = 0
    y = 0.1
    x_offset = 0.0
    while idx < n:
        x = x_offset
        while x < 1.0 and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 0.12
        y += 0.106
        x_offset = (1.0 - x_offset) * 0.5 # Toggle offset
        
    # Trim if more than 26 generated (though loop condition handles it)
    centers = centers[:n]
    
    # 2. Physics-based Relaxation (Growth and Repulsion)
    # This helps to spread circles out to allow them to grow larger
    velocities = np.zeros_like(centers)
    
    for step in range(2000):
        # Grow radii slightly
        radii += growth_rate
        
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    dist = 1e-9
                    diff = [np.random.rand(), np.random.rand()]
                
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    force_mag = repulsion_k * overlap
                    direction = diff / dist
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary repulsion
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            
            # Left
            if x - r < 0:
                forces[i, 0] += repulsion_k * (r - x)
            # Right
            if x + r > 1:
                forces[i, 0] -= repulsion_k * (x + r - 1)
            # Bottom
            if y - r < 0:
                forces[i, 1] += repulsion_k * (r - y)
            # Top
            if y + r > 1:
                forces[i, 1] -= repulsion_k * (y + r - 1)
        
        # Update velocities and positions
        velocities = velocities * damping + forces
        centers += velocities * 0.001
        
        # Clamp centers to valid range to prevent escaping
        centers[:, 0] = np.clip(centers[:, 0], 0.001, 0.999)
        centers[:, 1] = np.clip(centers[:, 1], 0.001, 0.999)

    # 3. CVXPY Optimization to maximize sum of radii given centers
    # We might run this a few times with slight perturbations if needed, 
    # but one good solve after relaxation is often strong.
    
    def optimize_radii(cent):
        r = cp.Variable(n)
        
        # Distances matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.linalg.norm(cent[i] - cent[j])
        
        constraints = []
        
        # Non-negative radii
        constraints.append(r >= 0)
        
        # Boundary constraints
        for i in range(n):
            x, y = cent[i]
            constraints.append(r[i] <= x)
            constraints.append(r[i] <= 1 - x)
            constraints.append(r[i] <= y)
            constraints.append(r[i] <= 1 - y)
            
        # Pairwise non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                # Add small epsilon for numerical stability
                constraints.append(r[i] + r[j] <= d - 1e-9)
                
        objective = cp.Maximize(cp.sum(r))
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use a reliable solver
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return r.value
            else:
                # Fallback to radii from simulation
                return radii
        except Exception:
            return radii

    # Run optimization
    optimal_radii = optimize_radii(centers)
    
    # 4. Final Validation and Correction
    # Ensure no negative radii and clip if any numerical errors
    optimal_radii = np.maximum(optimal_radii, 0)
    
    # Check for any potential overlap due to numerical precision and shrink slightly if needed
    # This is a safety step
    for _ in range(10):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                req_dist = optimal_radii[i] + optimal_radii[j]
                if dist < req_dist - 1e-12:
                    # Shrink radii equally
                    factor = (dist) / (req_dist + 1e-12) # slightly aggressive
                    optimal_radii[i] *= factor
                    optimal_radii[j] *= factor
    
    # Boundary check correction
    for i in range(n):
        x, y = centers[i]
        r = optimal_radii[i]
        optimal_radii[i] = min(r, x, 1-x, y, 1-y, 1e9) # Cap by boundary

    sum_radii = float(np.sum(optimal_radii))
    
    return centers, optimal_radii, sum_radii