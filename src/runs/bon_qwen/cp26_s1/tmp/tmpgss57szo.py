import numpy as np

def compute_overlap(centers, radii):
    n = len(centers)
    max_ov = 0.0
    for i in range(n):
        r_i = radii[i]
        x_i, y_i = centers[i]
        
        # Boundary checks
        if x_i < r_i:
            ov = r_i - x_i
            if ov > max_ov: max_ov = ov
        if x_i > 1 - r_i:
            ov = x_i - (1 - r_i)
            if ov > max_ov: max_ov = ov
        if y_i < r_i:
            ov = r_i - y_i
            if ov > max_ov: max_ov = ov
        if y_i > 1 - r_i:
            ov = y_i - (1 - r_i)
            if ov > max_ov: max_ov = ov
            
        # Circle-circle checks
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = (dx*dx + dy*dy)**0.5
            ov = r_i + radii[j] - dist
            if ov > max_ov:
                max_ov = ov
    return max_ov

def get_forces(centers, radii):
    n = len(centers)
    forces = np.zeros_like(centers)
    for i in range(n):
        r_i = radii[i]
        
        # Boundary repulsion
        if centers[i, 0] < r_i:
            forces[i, 0] += (r_i - centers[i, 0]) * 50
        elif centers[i, 0] > 1 - r_i:
            forces[i, 0] -= (centers[i, 0] - (1 - r_i)) * 50
            
        if centers[i, 1] < r_i:
            forces[i, 1] += (r_i - centers[i, 1]) * 50
        elif centers[i, 1] > 1 - r_i:
            forces[i, 1] -= (centers[i, 1] - (1 - r_i)) * 50
            
        # Inter-circle repulsion
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = dist_sq**0.5
            min_dist = r_i + radii[j]
            
            if dist < min_dist and dist > 1e-8:
                f_mag = (min_dist - dist) / dist_sq
                forces[i, 0] += dx * f_mag
                forces[i, 1] += dy * f_mag
                forces[j, 0] -= dx * f_mag
                forces[j, 1] -= dy * f_mag
            elif dist < 1e-8:
                # Coincident circles get a random kick to avoid singularity
                rand_dir = np.random.rand(2) * 2 - 1
                forces[i] += rand_dir
                forces[j] -= rand_dir
    return forces

def run_packing() -> tuple:
    np.random.seed(42)
    best_centers = None
    best_radii = None
    best_sum = 0.0
    n_circles = 26
    
    # Base hexagonal configuration
    base_centers = np.zeros((n_circles, 2))
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx < n_circles:
                x = 0.1 + j * 0.15 + (0.075 if i % 2 == 1 else 0)
                y = 0.1 + i * 0.16
                base_centers[idx] = [x, y]
                idx += 1
                
    # Run multiple trials with perturbations
    for trial in range(6):
        centers = base_centers.copy()
        if trial > 0:
            centers += np.random.uniform(-0.06, 0.06, centers.shape)
            centers = np.clip(centers, 0.02, 0.98)
            
        radii = np.ones(n_circles) * 0.04
        step = 0.04
        growth = 0.00025
        decay = 0.9996
        
        for _ in range(12000):
            forces = get_forces(centers, radii)
            centers += step * forces
            centers = np.clip(centers, 0, 1)
            
            max_ov = compute_overlap(centers, radii)
            
            # Adaptive growth strategy
            if max_ov < 1e-5:
                radii += growth
                growth *= 1.00015
            else:
                growth *= 0.994
                
            step *= decay
            if step < 1e-7:
                break
                
        curr_sum = np.sum(radii)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    # Final safety shrink to guarantee validity per checker tolerance
    final_ov = compute_overlap(best_centers, best_radii)
    if final_ov > 1e-9:
        best_radii -= final_ov + 1e-8
        best_radii = np.maximum(best_radii, 0)
        
    return best_centers, best_radii, np.sum(best_radii)