# sol_000359 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b4024b4) state=0b113f9c sum of radii=2.068959 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def calculate_overlap_penalty(centers, radii):
    """
    Calculate a smooth penalty for overlaps and boundary violations.
    Returns a positive value if constraints are violated, 0 otherwise.
    """
    n = centers.shape[0]
    penalty = 0.0
    eps = 1e-6
    
    # Pairwise overlaps
    # Using broadcasting for efficiency if N is small, otherwise loops
    # For N=26, loops are acceptable and memory efficient
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            sum_r = radii[i] + radii[j]
            overlap = sum_r - dist
            if overlap > 0:
                # Quadratic penalty for overlaps
                penalty += overlap ** 2
                
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Left/Right boundaries
        if x - r < 0: penalty += (x - r) ** 2
        if x + r > 1: penalty += (x + r - 1) ** 2
        # Top/Bottom boundaries
        if y - r < 0: penalty += (y - r) ** 2
        if y + r > 1: penalty += (y + r - 1) ** 2
        
    return penalty

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Number of circles
    N = 26
    
    # --- Stage 1: Heuristic Initialization and Growth ---
    
    # Initialize centers in a hexagonal pattern
    centers = np.zeros((N, 2))
    # Try to fit in a hexagonal lattice
    # Approximate radius to start with
    r_start = 0.09 
    
    # Simple hex packing logic
    count = 0
    y = r_start
    row_idx = 0
    while count < N:
        x = r_start + (row_idx % 2) * r_start # Offset odd rows
        while x <= 1 - r_start and count < N:
            centers[count] = [x, y]
            count += 1
            x += 2 * r_start
        y += r_start * np.sqrt(3)
        row_idx += 1
    
    # If we didn't fill enough, fill remaining randomly (should not happen with above logic for N=26)
    while count < N:
        centers[count] = [np.random.rand(), np.random.rand()]
        count += 1
        
    radii = np.full(N, r_start)
    
    # Growth Algorithm: Gradually increase radii and repel circles
    # This helps escape local minima of the grid
    num_growth_steps = 500
    growth_factor = 1.0002
    dt = 0.05 # Time step for position updates
    
    for step in range(num_growth_steps):
        # Increase radii
        radii *= growth_factor
        
        # Repulsion forces
        forces = np.zeros_like(centers)
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    dist = 1e-9
                    diff = np.random.rand(2) * 1e-9 # Avoid division by zero
                
                desired_dist = radii[i] + radii[j]
                if dist < desired_dist:
                    # Repulsive force proportional to overlap
                    overlap = desired_dist - dist
                    force_vec = (diff / dist) * overlap
                    forces[i] += force_vec
                    forces[j] -= force_vec
            
            # Boundary forces (push back inside)
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * 2.0
            # Right wall
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 2.0
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * 2.0
            # Top wall
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 2.0
        
        # Update positions
        centers += forces * dt
        
        # Clamp positions to [0, 1] to prevent wild excursions
        centers = np.clip(centers, 0.0, 1.0)
        
        # If penalty is low, we might want to increase growth rate slightly or keep stable
        # For simplicity, constant slow growth with repulsion works well
        
    # --- Stage 2: Local Optimization ---
    
    # Convert to a single vector for scipy optimization
    # Variables: [x1, y1, x2, y2, ..., r1, r2, ...]
    # Actually, optimizing positions and radii simultaneously can be tricky.
    # Let's optimize positions for fixed radii first? 
    # Or better: Maximize sum of radii with penalty.
    
    # Let's try to optimize the configuration to minimize overlap penalty 
    # while keeping radii roughly at the grown values, then try to increase radii.
    
    # First, relax positions to remove any residual overlaps from growth phase
    def objective_positions(vars):
        c = vars[:2*N].reshape(N, 2)
        return calculate_overlap_penalty(c, radii)
    
    initial_pos = centers.flatten()
    bounds_pos = [(0, 1)] * (2 * N)
    
    # Use L-BFGS-B for bound constraints
    res = opt.minimize(objective_positions, initial_pos, method='L-BFGS-B', bounds=bounds_pos, 
                       options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 1000})
    
    optimized_centers = res.x.reshape(N, 2)
    
    # Now, try to increase radii uniformly? 
    # Or allow them to vary? The problem asks to maximize sum of radii.
    # Let's try a final optimization where we treat radii as variables too.
    # But that increases dimensionality. 
    # Heuristic: The grown radii are likely close to optimal. 
    # We can try to scale them up slightly and re-relax.
    
    # Check current validity and sum
    current_sum = np.sum(radii)
    current_penalty = calculate_overlap_penalty(optimized_centers, radii)
    
    # If penalty is near 0, we can try to scale up radii
    if current_penalty < 1e-4:
        # Try to scale radii up by a small factor and re-optimize positions
        scale_factor = 1.005
        radii *= scale_factor
        
        # Re-optimize positions
        def objective_positions_scaled(vars):
            c = vars[:2*N].reshape(N, 2)
            return calculate_overlap_penalty(c, radii)
            
        res2 = opt.minimize(objective_positions_scaled, optimized_centers.flatten(), method='L-BFGS-B', 
                            bounds=bounds_pos, options={'ftol': 1e-9, 'maxiter': 500})
        optimized_centers = res2.x.reshape(N, 2)
        
        # Repeat scaling a few times if valid
        for _ in range(10):
            pen = calculate_overlap_penalty(optimized_centers, radii)
            if pen < 1e-5:
                radii *= 1.002
                res3 = opt.minimize(objective_positions_scaled, optimized_centers.flatten(), method='L-BFGS-B', 
                                    bounds=bounds_pos, options={'ftol': 1e-9, 'maxiter': 200})
                optimized_centers = res3.x.reshape(N, 2)
            else:
                break

    # Final cleanup: Ensure strict validity by clipping and reducing radius if necessary
    # Though optimizer should handle it, numerical errors might persist.
    
    # Verify and adjust if needed
    # If penalty is still high, we must reduce radii.
    penalty = calculate_overlap_penalty(optimized_centers, radii)
    if penalty > 1e-6:
        # Reduce radii proportionally to reduce penalty
        # This is a rough fix. Ideally we would do a proper optimization.
        # But for the purpose of returning a valid solution:
        # We can reduce all radii by a factor that clears overlaps.
        # Estimate max overlap
        max_overlap = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(optimized_centers[i] - optimized_centers[j])
                req_dist = radii[i] + radii[j]
                if dist < req_dist:
                    max_overlap = max(max_overlap, req_dist - dist)
            
            x, y = optimized_centers[i]
            r = radii[i]
            if x < r: max_overlap = max(max_overlap, r - x)
            if x > 1 - r: max_overlap = max(max_overlap, x - (1 - r))
            if y < r: max_overlap = max(max_overlap, r - y)
            if y > 1 - r: max_overlap = max(max_overlap, y - (1 - r))
            
        if max_overlap > 0:
            # Reduce radii to eliminate max_overlap. 
            # Since reducing r reduces req_dist by 2*dr (for pairs) and 1*dr (for bounds),
            # we can approximate.
            radii -= max_overlap * 1.1 # Over-reduce slightly to be safe
            
    # Final check for negative radii
    radii = np.maximum(radii, 0)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    return optimized_centers, radii, sum_radii
