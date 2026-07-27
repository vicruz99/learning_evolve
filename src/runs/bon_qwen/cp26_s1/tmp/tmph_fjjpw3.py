import numpy as np
from scipy.optimize import minimize
import math


def run_packing() -> tuple:
    n = 26
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Try multiple strategies and seeds
    for seed in range(30):
        np.random.seed(seed)
        
        # Try different initializations
        for init_type in range(3):
            centers, radii = initialize_packing(n, init_type)
            
            # Phase 1: Physics-based relaxation
            centers, radii = physics_relax(centers, radii, n, max_iter=3000)
            
            # Phase 2: Try scipy optimization
            centers, radii = scipy_optimize(centers, radii, n)
            
            # Validate and track best
            current_sum = np.sum(radii)
            if is_valid_packing(centers, radii, n) and current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
    
    # Final cleanup
    return best_centers, best_radii, best_sum


def initialize_packing(n, init_type):
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    if init_type == 0:
        # Hexagonal grid
        r_init = 0.065
        idx = 0
        y = r_init
        row = 0
        while idx < n:
            x = r_init
            if row % 2 == 1:
                x = r_init + r_init * math.sqrt(3) / 2
            while idx < n and x <= 1 - r_init:
                centers[idx] = [x, y]
                radii[idx] = r_init
                idx += 1
                x += 2 * r_init
            y += r_init * math.sqrt(3)
            row += 1
        # Fill remaining
        while idx < n:
            centers[idx] = [0.5, 0.5]
            radii[idx] = r_init
            idx += 1
            
    elif init_type == 1:
        # Grid pattern with perturbation
        rows = 6
        cols = math.ceil(n / rows)
        r_init = 0.06
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                centers[idx] = [x, y]
                radii[idx] = r_init
                idx += 1
        centers += np.random.uniform(-0.02, 0.02, (n, 2))
        
    else:
        # Random initialization with repulsion
        radii[:] = 0.05
        for i in range(n):
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                ok = True
                for j in range(i):
                    dist = math.sqrt((x - centers[j, 0])**2 + (y - centers[j, 1])**2)
                    if dist < radii[j] + 0.05 + 0.01:
                        ok = False
                        break
                if ok:
                    centers[i] = [x, y]
                    placed = True
                attempts += 1
            if not placed:
                centers[i] = [0.5, 0.5]
    
    return centers, radii


def physics_relax(centers, radii, n, max_iter=3000):
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iter):
        forces = np.zeros((n, 2))
        cooling = max(0.0001, 0.01 * (1 - iteration / max_iter))
        
        # Calculate repulsion forces between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-10
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Strong repulsion for overlaps
                    overlap = min_dist - dist
                    force_mag = overlap * 50 / (dist + 0.001)
                elif dist < min_dist * 1.1:
                    # Weaker repulsion for near overlaps
                    overlap = min_dist * 1.1 - dist
                    force_mag = overlap * 2 / (dist + 0.001)
                else:
                    force_mag = 0
                
                if force_mag > 0 and dist > 0:
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            if x - r < 0:
                forces[i, 0] += (r - x) * 100
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 100
            if y - r < 0:
                forces[i, 1] += (r - y) * 100
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 100
        
        # Apply forces with momentum
        centers += forces * cooling
        
        # Clip to valid range
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i] + 1e-8, 1 - radii[i] - 1e-8)
            centers[i, 1] = np.clip(centers[i, 1], radii[i] + 1e-8, 1 - radii[i] - 1e-8)
        
        # Gradually increase radii
        expansion_rate = 0.0005 * (1 - iteration / max_iter * 0.5)
        for i in range(n):
            min_gap = 1.0
            
            # Check gap to other circles
            for j in range(n):
                if i != j:
                    dist = math.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                                   (centers[i, 1] - centers[j, 1])**2)
                    min_gap = min(min_gap, (dist - radii[j]) / 2)
            
            # Check gap to boundary
            min_gap = min(min_gap, centers[i, 0] - radii[i], 
                         1 - centers[i, 0] - radii[i],
                         centers[i, 1] - radii[i],
                         1 - centers[i, 1] - radii[i])
            
            if min_gap > 1e-6:
                radii[i] += min(min_gap * expansion_rate, 0.001)
    
    return centers, radii


def scipy_optimize(centers, radii, n):
    """Use scipy to fine-tune the packing"""
    params = np.concatenate([centers.flatten(), radii])
    
    # Bounds: positions in [0,1], radii in [0.01, 0.5]
    bounds = [(0.001, 0.999)] * (2 * n) + [(0.01, 0.5)] * n
    
    result = minimize(
        objective_penalty,
        params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
    )
    
    centers_opt = result.x[:2*n].reshape(n, 2)
    radii_opt = result.x[2*n:]
    
    # Ensure validity
    radii_opt = np.clip(radii_opt, 1e-8, 0.5)
    for i in range(n):
        centers_opt[i, 0] = np.clip(centers_opt[i, 0], radii_opt[i] + 1e-9, 1 - radii_opt[i] - 1e-9)
        centers_opt[i, 1] = np.clip(centers_opt[i, 1], radii_opt[i] + 1e-9, 1 - radii_opt[i] - 1e-9)
    
    return centers_opt, radii_opt


def objective_penalty(params):
    n = 26
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    obj = -np.sum(radii)
    
    # Boundary penalties
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        penalty = 0
        if x - r < 0:
            penalty += (r - x) ** 2
        if x + r > 1:
            penalty += (x + r - 1) ** 2
        if y - r < 0:
            penalty += (r - y) ** 2
        if y + r > 1:
            penalty += (y + r - 1) ** 2
        obj += 5000 * penalty
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                obj += 5000 * overlap ** 2
    
    return obj


def is_valid_packing(centers, radii, n):
    """Check if packing is valid"""
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    
    for i in range(n):
        if radii[i] < 0:
            return False
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-10 or x + r > 1 + 1e-10:
            return False
        if y - r < -1e-10 or y + r > 1 + 1e-10:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-10:
                return False
    
    return True