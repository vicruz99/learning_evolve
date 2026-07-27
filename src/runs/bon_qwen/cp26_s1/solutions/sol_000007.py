# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=6130cc24 sum of radii=0.514899 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np


def validate_internal(centers, radii):
    """Internal validation with stricter tolerance"""
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-14 or x + r > 1 + 1e-14 or y - r < -1e-14 or y + r > 1 + 1e-14:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-14:
                return False
    return True


def optimize_packing(centers, radii, max_epochs=4000):
    """Iterative growing and relaxation optimization"""
    n = len(radii)
    
    for epoch in range(max_epochs):
        # Adaptive growth rate - slower as we converge
        progress = epoch / max_epochs
        growth = 0.000008 * max(0.05, 1.0 - progress * 0.95)
        
        # Grow all radii
        radii += growth
        
        # Relax positions to resolve overlaps
        for step in range(4000):
            forces = np.zeros_like(centers)
            max_f = 0.0
            
            # Circle-circle repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy + 1e-24)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Nonlinear repulsion for stronger push when deeply overlapped
                        f = overlap * 800.0
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                        if f > max_f:
                            max_f = f
            
            # Boundary repulsion
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                
                if x - r < 0:
                    f = (r - x) * 800.0
                    forces[i, 0] += f
                    if f > max_f:
                        max_f = f
                if x + r > 1:
                    f = (x + r - 1) * 800.0
                    forces[i, 0] -= f
                    if f > max_f:
                        max_f = f
                if y - r < 0:
                    f = (r - y) * 800.0
                    forces[i, 1] += f
                    if f > max_f:
                        max_f = f
                if y + r > 1:
                    f = (y + r - 1) * 800.0
                    forces[i, 1] -= f
                    if f > max_f:
                        max_f = f
            
            # Adaptive step size
            alpha = 0.015 / (1.0 + max_f * 0.0005)
            centers = centers + forces * alpha
            
            # Project to feasible region
            for i in range(n):
                centers[i, 0] = max(radii[i] + 1e-13, min(1.0 - radii[i] - 1e-13, centers[i, 0]))
                centers[i, 1] = max(radii[i] + 1e-13, min(1.0 - radii[i] - 1e-13, centers[i, 1]))
            
            # Early stopping if converged
            if max_f < 1e-18:
                break
        
        # Check if configuration is valid
        valid = validate_internal(centers, radii)
        
        if not valid:
            # Back off
            radii = radii - growth * 2.0
            radii = np.maximum(radii, 0.001)
    
    return centers, radii


def create_hexagonal_config(n, row_counts, y_spacing, x_spacing, y_start, x_start):
    """Create a hexagonal grid configuration"""
    centers = np.zeros((n, 2))
    idx = 0
    
    for row_i, count in enumerate(row_counts):
        y = y_start + row_i * y_spacing
        x_offset = x_start if row_i % 2 == 0 else x_start + x_spacing / 2.0
        for col_j in range(count):
            x = x_offset + col_j * x_spacing
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
    
    return centers


def run_packing():
    n = 26
    
    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.full(n, 0.01)
    
    # Define multiple initial configurations to try
    configs = []
    
    # Config 1: Rows [5,5,5,5,4,2]
    configs.append(create_hexagonal_config(n, [5, 5, 5, 5, 4, 2], 
                                          y_spacing=0.112, x_spacing=0.133, 
                                          y_start=0.06, x_start=0.035))
    
    # Config 2: Rows [6,5,5,5,4,1]
    configs.append(create_hexagonal_config(n, [6, 5, 5, 5, 4, 1], 
                                          y_spacing=0.115, x_spacing=0.130, 
                                          y_start=0.055, x_start=0.028))
    
    # Config 3: Rows [5,5,5,5,5,1]
    configs.append(create_hexagonal_config(n, [5, 5, 5, 5, 5, 1], 
                                          y_spacing=0.118, x_spacing=0.135, 
                                          y_start=0.055, x_start=0.030))
    
    # Config 4: Rows [4,5,5,5,5,2]
    configs.append(create_hexagonal_config(n, [4, 5, 5, 5, 5, 2], 
                                          y_spacing=0.115, x_spacing=0.138, 
                                          y_start=0.06, x_start=0.05))
    
    # Config 5: Rows [5,5,5,5,3,3]
    configs.append(create_hexagonal_config(n, [5, 5, 5, 5, 3, 3], 
                                          y_spacing=0.113, x_spacing=0.140, 
                                          y_start=0.058, x_start=0.040))
    
    # Config 6: Rows [4,5,5,5,4,3]
    configs.append(create_hexagonal_config(n, [4, 5, 5, 5, 4, 3], 
                                          y_spacing=0.112, x_spacing=0.142, 
                                          y_start=0.060, x_start=0.055))
    
    # Config 7: Rows [5,5,5,4,4,3]
    configs.append(create_hexagonal_config(n, [5, 5, 5, 4, 4, 3], 
                                          y_spacing=0.118, x_spacing=0.145, 
                                          y_start=0.055, x_start=0.035))
    
    # Config 8: Rows [3,5,5,5,5,3]
    configs.append(create_hexagonal_config(n, [3, 5, 5, 5, 5, 3], 
                                          y_spacing=0.113, x_spacing=0.150, 
                                          y_start=0.058, x_start=0.060))
    
    # Config 9: Rows [5,5,5,5,5,1] with tighter spacing
    configs.append(create_hexagonal_config(n, [5, 5, 5, 5, 5, 1], 
                                          y_spacing=0.110, x_spacing=0.128, 
                                          y_start=0.065, x_start=0.038))
    
    # Config 10: Rows [6,5,5,5,5]
    configs.append(create_hexagonal_config(n, [6, 5, 5, 5, 5], 
                                          y_spacing=0.125, x_spacing=0.135, 
                                          y_start=0.060, x_start=0.030))
    
    for config_idx, init_centers in enumerate(configs):
        centers = init_centers.copy()
        radii = np.full(n, 0.003)
        
        # Run optimization
        centers, radii = optimize_packing(centers, radii, max_epochs=4000)
        
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Final cleanup: ensure strict validity with margin
    for i in range(n):
        best_centers[i, 0] = max(best_radii[i] + 1e-10, min(1.0 - best_radii[i] - 1e-10, best_centers[i, 0]))
        best_centers[i, 1] = max(best_radii[i] + 1e-10, min(1.0 - best_radii[i] - 1e-10, best_centers[i, 1]))
    
    return best_centers, best_radii, np.sum(best_radii)
