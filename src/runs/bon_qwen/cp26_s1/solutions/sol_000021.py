# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=735790f5 sum of radii=1.746711 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_forces(centers, radii):
    """Compute repulsive forces for circle packing relaxation."""
    n = len(radii)
    forces = np.zeros_like(centers)
    strength = 3.0
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[j] - centers[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                diff = np.random.rand(2) - 0.5
                dist = np.linalg.norm(diff)
                
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                f_magnitude = overlap**2 * strength
                dir_vec = diff / dist
                forces[i] -= f_magnitude * dir_vec
                forces[j] += f_magnitude * dir_vec
                
    # Boundary repulsion
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        if x < r:
            forces[i, 0] += (r - x)**2 * strength
        elif x > 1 - r:
            forces[i, 0] -= (x - (1 - r))**2 * strength
            
        if y < r:
            forces[i, 1] += (r - y)**2 * strength
        elif y > 1 - r:
            forces[i, 1] -= (y - (1 - r))**2 * strength
            
    return forces

def is_valid(centers, radii, tol=1e-9):
    """Check if packing satisfies all constraints."""
    n = len(radii)
    if np.any(radii < 0):
        return False
        
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
            
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def run_packing() -> tuple:
    np.random.seed(42)
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Prepare multiple initializations
    inits = []
    
    # 1. Dense Grid
    c1 = np.zeros((n, 2))
    idx = 0
    for r in range(5):
        for c in range(5):
            if idx < n:
                c1[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                idx += 1
    inits.append(c1)
    
    # 2. Random Scattered
    c2 = np.random.uniform(0.15, 0.85, (n, 2))
    inits.append(c2)
    
    # 3. Hexagonal-ish
    c3 = np.zeros((n, 2))
    idx = 0
    row_counts = [5, 6, 5, 6, 4]
    y_pos = 0.12
    for rc in row_counts:
        x_start = 0.5 - (rc - 1) * 0.08
        for i in range(rc):
            if idx < n:
                c3[idx] = [x_start + i * 0.16, y_pos]
                idx += 1
        y_pos += 0.135
    inits.append(c3)
    
    for init_centers in inits:
        centers = init_centers.copy()
        radii = np.ones(n) * 0.025
        
        dr = 0.002
        lr = 0.1
        max_steps = 2500
        
        for step in range(max_steps):
            # Relaxation step
            forces = compute_forces(centers, radii)
            centers += forces * lr
            centers = np.clip(centers, 0.0, 1.0)
            
            # Grow radii with slight random variation to encourage diverse sizes
            radii += dr + np.random.uniform(-0.0005, 0.0005, n)
            
            # Periodic jitter to escape local minima
            if step % 120 == 0 and step > 0:
                centers += np.random.randn(*centers.shape) * 0.015
                centers = np.clip(centers, 0.0, 1.0)
                
            # Validation check
            if is_valid(centers, radii):
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
                    # If valid, we can try to grow slightly faster
                    dr *= 1.005
                    lr *= 0.998
            else:
                # Violation: backtrack radii and reduce step size
                radii -= dr * 1.8
                dr *= 0.96
                lr *= 1.05
                
            # Convergence check
            if dr < 1e-7:
                break
                
        # Final high-resolution polish for this run
        for _ in range(400):
            forces = compute_forces(centers, radii)
            centers += forces * 0.05
            centers = np.clip(centers, 0.0, 1.0)
            
        if is_valid(centers, radii):
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # Ensure non-negative radii
    if best_radii is not None:
        best_radii = np.maximum(best_radii, 0.0)
        
    return best_centers, best_radii, best_sum
