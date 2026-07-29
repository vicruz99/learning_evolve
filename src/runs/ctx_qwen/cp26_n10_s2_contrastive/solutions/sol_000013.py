# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b4d6f452) state=aea429bf sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective_func(x):
    """Negative sum of radii (we minimize)"""
    radii = x[2*N:]
    return -np.sum(radii)

def boundary_left(x, i):
    return x[2*i] - x[2*N + i]

def boundary_right(x, i):
    return 1.0 - x[2*i] - x[2*N + i]

def boundary_bottom(x, i):
    return x[2*i + 1] - x[2*N + i]

def boundary_top(x, i):
    return 1.0 - x[2*i + 1] - x[2*N + i]

def non_overlap(x, indices):
    """Squared distance constraint: dist^2 >= (r_i + r_j)^2"""
    i, j = indices
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist_sq = dx*dx + dy*dy
    r_sum = x[2*N+i] + x[2*N+j]
    return dist_sq - r_sum * r_sum

def create_initial_placement(n, seed=0):
    """Create a hexagonal-like initial placement that is valid (no overlaps)"""
    rng = np.random.RandomState(seed)
    
    r_init = 0.075
    
    centers = []
    radii = []
    
    # Hexagonal pattern: 6, 5, 6, 5, 4 = 26 circles
    row_counts = [6, 5, 6, 5, 4]
    
    y_spacing = np.sqrt(3) * r_init
    x_spacing = 2 * r_init
    
    y = r_init + 0.015
    for row_idx, ncols in enumerate(row_counts):
        x_start = r_init + 0.015
        if row_idx % 2 == 1:
            x_start += r_init * 0.7
        
        for col in range(ncols):
            x = x_start + col * x_spacing
            if x + r_init <= 1.0:
                centers.append([x, y])
                radii.append(r_init)
        
        y += y_spacing
    
    # Fill remaining circles if we have fewer than 26
    while len(centers) < n:
        placed = False
        for attempt in range(300):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            valid = True
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy < (2.2 * r_init) ** 2:
                    valid = False
                    break
            if valid:
                centers.append([x, y])
                radii.append(r_init)
                placed = True
                break
        if not placed:
            # Try to find any gap
            best_pos = None
            best_dist = 0
            for attempt in range(500):
                x = rng.uniform(0.15, 0.85)
                y = rng.uniform(0.15, 0.85)
                min_dist = 1.0
                for cx, cy in centers:
                    d = np.sqrt((x-cx)**2 + (y-cy)**2)
                    min_dist = min(min_dist, d)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_pos = (x, y)
            if best_pos and best_dist > r_init:
                centers.append(list(best_pos))
                radii.append(r_init * 0.6)
            else:
                centers.append([0.5, 0.5])
                radii.append(r_init * 0.3)
    
    centers = np.array(centers[:n])
    radii = np.array(radii[:n])
    
    return np.concatenate([centers.flatten(), radii])

def validate_packing_internal(centers, radii, n):
    """Internal validation of packing"""
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-10 or x + r > 1 + 1e-10:
            return False
        if y - r < -1e-10 or y + r > 1 + 1e-10:
            return False
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            if dist_sq < (r_sum - 1e-10) ** 2:
                return False
    return True

def run_packing():
    n = N
    
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Try multiple initial placements with different seeds
    for seed in range(15):
        x0 = create_initial_placement(n, seed=seed)
        
        # Define bounds: x,y in [0,1], r in [1e-6, 0.5]
        bounds = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
        
        # Build constraints
        constraints = []
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': boundary_left, 'args': (i,)})
            constraints.append({'type': 'ineq', 'fun': boundary_right, 'args': (i,)})
            constraints.append({'type': 'ineq', 'fun': boundary_bottom, 'args': (i,)})
            constraints.append({'type': 'ineq', 'fun': boundary_top, 'args': (i,)})
        
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': non_overlap, 'args': ((i, j),)})
        
        try:
            result = minimize(
                objective_func, 
                x0, 
                bounds=bounds, 
                constraints=constraints, 
                method='SLSQP', 
                options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
            )
            
            centers = result.x[:2*n].reshape(n, 2)
            radii = result.x[2*n:]
            
            # Check if this is valid
            if validate_packing_internal(centers, radii, n):
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
                    # If we found a good solution, try to refine it
                    if current_sum > 2.5:
                        # Run another optimization from this good solution
                        x0_refine = np.concatenate([centers.flatten(), radii])
                        result2 = minimize(
                            objective_func,
                            x0_refine,
                            bounds=bounds,
                            constraints=constraints,
                            method='SLSQP',
                            options={'maxiter': 10000, 'ftol': 1e-15, 'disp': False}
                        )
                        centers2 = result2.x[:2*n].reshape(n, 2)
                        radii2 = result2.x[2*n:]
                        
                        if validate_packing_internal(centers2, radii2, n):
                            current_sum2 = np.sum(radii2)
                            if current_sum2 > best_sum:
                                best_sum = current_sum2
                                best_centers = centers2.copy()
                                best_radii = radii2.copy()
                                
        except Exception:
            continue
    
    if best_centers is None:
        # Fallback: create a simple valid packing
        x0 = create_initial_placement(n, seed=42)
        centers = x0[:2*n].reshape(n, 2)
        radii = x0[2*n:]
        return centers, radii, float(np.sum(radii))
    
    return best_centers, best_radii, float(best_sum)
