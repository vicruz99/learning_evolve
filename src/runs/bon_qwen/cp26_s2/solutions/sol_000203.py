# sol_000203 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=4bd2ccfe sum of radii=1.511095 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_forces_and_update(centers, radii, dt):
    """
    Compute repulsive forces and update centers to resolve overlaps and boundary violations.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    repulsion_k = 1.5
    boundary_k = 3.0
    
    # Boundary forces
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Left wall
        if x - r < 0:
            forces[i, 0] += boundary_k * (r - x)
        # Right wall
        if x + r > 1:
            forces[i, 0] -= boundary_k * (x + r - 1)
        # Bottom wall
        if y - r < 0:
            forces[i, 1] += boundary_k * (r - y)
        # Top wall
        if y + r > 1:
            forces[i, 1] -= boundary_k * (y + r - 1)
            
    # Inter-circle repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist and dist > 1e-12:
                overlap = min_dist - dist
                nx = dx / dist
                ny = dy / dist
                # Push apart proportionally to inverse radius (smaller circles move more)
                w_i = 1.0 / radii[i]
                w_j = 1.0 / radii[j]
                w_sum = w_i + w_j
                
                forces[i, 0] -= repulsion_k * overlap * nx * (w_i / w_sum)
                forces[i, 1] -= repulsion_k * overlap * ny * (w_i / w_sum)
                forces[j, 0] += repulsion_k * overlap * nx * (w_j / w_sum)
                forces[j, 1] += repulsion_k * overlap * ny * (w_j / w_sum)
                
    # Update positions
    centers += forces * dt
    
    # Clamp strictly to valid range to prevent drift
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
        
    return centers

def tighten_radii(centers, radii):
    """
    Set each radius to the maximum feasible value given current positions, slightly shrunk for safety.
    """
    n = centers.shape[0]
    safety = 0.9992
    
    for i in range(n):
        max_r = np.min([centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1]])
        for j in range(n):
            if i == j:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 0:
                max_r = np.min([max_r, (dist - radii[j]) / 2.0])
        radii[i] = np.max([max_r, 0.0]) * safety
    return radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # 1. Hexagonal Lattice Initialization
    centers = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        # Hexagonal offset
        offset = (0.5 if row % 2 == 1 else 0.0)
        x_start = 0.05 + offset * 0.2
        y_val = 0.05 + row * 0.19
        
        # Estimate how many fit in this row
        count = int(np.floor((0.9 - x_start) / 0.2)) + 1
        for col in range(count):
            if idx >= n:
                break
            centers[idx, 0] = x_start + col * 0.2
            centers[idx, 1] = y_val
            idx += 1
        row += 1
        
    # Adjust initial radii to a feasible small value
    radii = np.full(n, 0.06)
    
    # 2. Iterative Optimization
    max_iter = 3000
    base_dt = 0.08
    grow_rate = 1.0004
    
    for step in range(max_iter):
        # Cooling schedule for step size
        dt = base_dt * (0.9995 ** step)
        
        # Expand radii
        radii *= (1.0 + 0.0003)
        
        # Resolve overlaps and boundaries
        centers = compute_forces_and_update(centers, radii, dt)
        
        # Local tightening every 10 steps
        if step % 10 == 0:
            radii = tighten_radii(centers, radii)
            
        # Random perturbation to escape local minima
        if step % 200 == 0 and step > 500:
            perturbation = np.random.normal(0, 0.005, size=centers.shape)
            centers += perturbation
            # Re-clamp
            centers = np.clip(centers, 0.01, 0.99)
            radii = tighten_radii(centers, radii)
            
    # 3. Final Tightening & Validation Correction
    # Run tightening multiple times to propagate constraints
    for _ in range(50):
        radii = tighten_radii(centers, radii)
        centers = compute_forces_and_update(centers, radii, 0.05)
        
    # Final safety shrink to guarantee validation passes
    radii *= 0.9995
    
    # Ensure absolute validity
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        centers[i, 0] = np.clip(x, r, 1.0 - r)
        centers[i, 1] = np.clip(y, r, 1.0 - r)
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
