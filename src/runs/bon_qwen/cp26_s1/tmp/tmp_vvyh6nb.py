import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_penalty(centers, radii):
    """Compute constraint violation penalty"""
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Left boundary
        if x - r < 0:
            penalty += 1e8 * (x - r) ** 2
        # Right boundary
        if x + r > 1:
            penalty += 1e8 * (x + r - 1) ** 2
        # Bottom boundary
        if y - r < 0:
            penalty += 1e8 * (y - r) ** 2
        # Top boundary
        if y + r > 1:
            penalty += 1e8 * (y + r - 1) ** 2
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                gap = min_dist - dist
                penalty += 1e8 * gap ** 2
    
    # Negative radius violations
    for i in range(n):
        if radii[i] < 0:
            penalty += 1e8 * radii[i] ** 2
    
    return penalty

def objective(params):
    """Objective: negative sum of radii + penalty for violations"""
    n = N_CIRCLES
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    obj = -np.sum(radii)
    penalty = compute_penalty(centers, radii)
    
    return obj + penalty

def hexagonal_init(r_base=0.09):
    """Initialize circles in a hexagonal pattern"""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.full(n, r_base)
    
    spacing_x = 2 * r_base
    spacing_y = r_base * np.sqrt(3)
    
    idx = 0
    row = 0
    while idx < n:
        col = 0
        offset = r_base if row % 2 == 1 else 0
        while idx < n:
            x = r_base + col * spacing_x + offset
            y = r_base + row * spacing_y
            
            if x + r_base <= 1 and y + r_base <= 1:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    
    return centers, radii

def grid_init(r_base=0.09):
    """Initialize circles in a grid pattern"""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.full(n, r_base)
    
    spacing = 2 * r_base
    idx = 0
    row = 0
    while idx < n:
        col = 0
        while idx < n:
            x = r_base + col * spacing
            y = r_base + row * spacing
            
            if x + r_base <= 1 and y + r_base <= 1:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    
    return centers, radii

def random_init(seed=0):
    """Initialize circles randomly"""
    n = N_CIRCLES
    rng = np.random.RandomState(seed)
    centers = rng.rand(n, 2) * 0.6 + 0.2
    radii = 0.05 + 0.05 * rng.rand(n)
    return centers, radii

def mixed_init():
    """Initialize with larger circles in corners and smaller in center"""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 4 corner circles
    r_corner = 0.22
    corners = [(r_corner, r_corner), (1 - r_corner, r_corner), 
               (r_corner, 1 - r_corner), (1 - r_corner, 1 - r_corner)]
    for i, (x, y) in enumerate(corners):
        centers[i] = [x, y]
        radii[i] = r_corner
    
    # Fill remaining with smaller circles in center
    remaining = n - 4
    r_small = 0.06
    idx = 4
    row = 0
    while idx < n:
        col = 0
        while idx < n:
            x = 0.3 + col * 2 * r_small
            y = 0.3 + row * 2 * r_small
            if x + r_small <= 0.7 and y + r_small <= 0.7:
                centers[idx] = [x, y]
                radii[idx] = r_small
                idx += 1
            col += 1
        row += 1
    
    return centers, radii

def cleanup(centers, radii):
    """Ensure all constraints are satisfied"""
    n = len(radii)
    
    # Clip radii to non-negative
    radii = np.maximum(radii, 0)
    
    # Adjust centers to respect boundaries
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
    
    # Iteratively reduce radii to eliminate overlaps
    for iteration in range(200):
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j]:
                    overlap = radii[i] + radii[j] - dist
                    max_overlap = max(max_overlap, overlap)
                    # Reduce radii proportionally
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        radii[i] -= overlap * radii[i] / total_r
                        radii[j] -= overlap * radii[j] / total_r
        
        radii = np.maximum(radii, 0)
        
        # Re-adjust centers
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
        
        if max_overlap < 1e-10:
            break
    
    return centers, radii

def expand_radii(centers, radii):
    """Try to expand radii as much as possible"""
    n = len(radii)
    
    for iteration in range(50):
        expanded = False
        for i in range(n):
            # Find maximum possible radius for circle i
            max_r = min(centers[i, 0], 1 - centers[i, 0],
                       centers[i, 1], 1 - centers[i, 1])
            
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - radii[j])
            
            max_r = max(0, max_r)
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                expanded = True
        
        if not expanded:
            break
    
    return centers, radii

def run_packing():
    n = N_CIRCLES
    
    best_sum = -1
    best_centers = None
    best_radii = None
    
    # Generate initial configurations
    inits = []
    
    # Hexagonal patterns with different base radii
    for r in [0.07, 0.08, 0.09, 0.10, 0.11]:
        try:
            c, r_arr = hexagonal_init(r)
            if len([x for x in c if any(v != 0 for v in x)]) >= n:
                inits.append((c, r_arr))
        except:
            pass
    
    # Grid patterns
    for r in [0.07, 0.08, 0.09, 0.10]:
        try:
            c, r_arr = grid_init(r)
            if len([x for x in c if any(v != 0 for v in x)]) >= n:
                inits.append((c, r_arr))
        except:
            pass
    
    # Random initializations
    for seed in range(20):
        inits.append(random_init(seed))
    
    # Mixed initialization
    inits.append(mixed_init())
    
    # Optimize each initialization
    for centers, radii in inits:
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Bounds: centers in [0,1], radii in [0, 0.5]
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        # Optimize with Nelder-Mead
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 200000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
        except:
            continue
        
        centers_opt = result.x[:2 * n].reshape(n, 2)
        radii_opt = result.x[2 * n:]
        
        # Cleanup
        centers_opt, radii_opt = cleanup(centers_opt, radii_opt)
        
        # Try to expand radii
        centers_opt, radii_opt = expand_radii(centers_opt, radii_opt)
        
        current_sum = np.sum(radii_opt)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    # Final cleanup
    best_centers, best_radii = cleanup(best_centers, best_radii)
    
    return best_centers, best_radii, np.sum(best_radii)