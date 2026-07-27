import numpy as np
from scipy.optimize import minimize


def make_init_grid(n, spacing=0.16, radius=0.08):
    centers = np.zeros((n, 2))
    radii = np.ones(n) * radius
    idx = 0
    for i in range(7):
        for j in range(5):
            if idx >= n:
                break
            x = 0.12 + i * spacing
            y = 0.12 + j * spacing
            centers[idx] = [x, y]
            idx += 1
        if idx >= n:
            break
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1
    return centers, radii


def make_init_hex(n, radius=0.08):
    centers = np.zeros((n, 2))
    radii = np.ones(n) * radius
    idx = 0
    row = 0
    while idx < n:
        y = 0.08 + row * 0.14
        if y > 0.95:
            break
        offset = 0.07 if row % 2 == 1 else 0.0
        x_start = 0.08 + offset
        for j in range(5):
            if idx >= n:
                break
            x = x_start + j * 0.18
            if x <= 0.95:
                centers[idx] = [x, y]
                idx += 1
        row += 1
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1
    return centers, radii


def make_init_corners(n):
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.06
    
    # 4 corner circles
    rc = 0.2
    centers[0] = [rc, rc]
    centers[1] = [1 - rc, rc]
    centers[2] = [rc, 1 - rc]
    centers[3] = [1 - rc, 1 - rc]
    radii[0:4] = rc
    
    # 4 edge mid circles
    re = 0.12
    centers[4] = [0.5, re]
    centers[5] = [re, 0.5]
    centers[6] = [1 - re, 0.5]
    centers[7] = [0.5, 1 - re]
    radii[4:8] = re
    
    # Center
    centers[8] = [0.5, 0.5]
    radii[8] = 0.08
    
    # Fill rest in grid
    idx = 9
    for i in range(6):
        for j in range(4):
            if idx >= n:
                break
            centers[idx] = [0.15 + i * 0.14, 0.15 + j * 0.18]
            radii[idx] = 0.05
            idx += 1
        if idx >= n:
            break
    while idx < n:
        centers[idx] = [np.random.rand() * 0.6 + 0.2, np.random.rand() * 0.6 + 0.2]
        idx += 1
    
    return centers, radii


def compute_constraints(x, n):
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    result = []
    
    # Boundary constraints
    for i in range(n):
        result.append(centers[i, 0] - radii[i])
        result.append(1 - centers[i, 0] - radii[i])
        result.append(centers[i, 1] - radii[i])
        result.append(1 - centers[i, 1] - radii[i])
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            result.append(dist - radii[i] - radii[j])
    
    return np.array(result)


def optimize_from(centers, radii, n, max_iter=5000):
    x0 = np.concatenate([centers.flatten(), radii])
    
    def objective(x):
        return -np.sum(x[2 * n:])
    
    def cons_fun(x):
        return compute_constraints(x, n)
    
    constraints = {'type': 'ineq', 'fun': cons_fun}
    
    bounds = [(0.001, 0.999)] * (2 * n) + [(0.0001, 0.5)] * n
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'disp': False}
    )
    
    return result


def force_refine(centers, radii, n, steps=2000):
    """Force-based refinement to further optimize packing"""
    centers = centers.copy()
    radii = radii.copy()
    
    lr = 0.003
    repulsion = 200.0
    
    for step in range(steps):
        # Grow radii slowly
        growth = 1.0 + 0.00008
        radii *= growth
        
        for i in range(n):
            fx, fy = 0.0, 0.0
            
            # Boundary forces
            for d in range(2):
                if centers[i, d] < radii[i] + 1e-10:
                    force = repulsion * (radii[i] - centers[i, d])
                    if d == 0:
                        fx += force
                    else:
                        fy += force
                if centers[i, d] > 1 - radii[i] - 1e-10:
                    force = repulsion * (centers[i, d] - (1 - radii[i]))
                    if d == 0:
                        fx -= force
                    else:
                        fy -= force
            
            # Overlap repulsion
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff * diff))
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    f = repulsion * overlap / dist
                    fx += diff[0] * f
                    fy += diff[1] * f
            
            # Update position
            centers[i, 0] += lr * fx
            centers[i, 1] += lr * fy
            
            # Clamp to valid range
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
        
        if step > 0 and step % 500 == 0:
            lr *= 0.85
    
    return centers, radii


def run_packing() -> tuple:
    n = 26
    
    best_centers = None
    best_radii = None
    best_sum = -np.inf
    
    # Strategy 1: Grid initialization
    centers, radii = make_init_grid(n, spacing=0.15, radius=0.07)
    result = optimize_from(centers, radii, n, max_iter=3000)
    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = np.maximum(result.x[2 * n:], 1e-10)
    s = np.sum(radii_opt)
    if s > best_sum:
        best_sum = s
        best_centers = centers_opt.copy()
        best_radii = radii_opt.copy()
    
    # Strategy 2: Hexagonal initialization
    centers, radii = make_init_hex(n, radius=0.07)
    result = optimize_from(centers, radii, n, max_iter=3000)
    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = np.maximum(result.x[2 * n:], 1e-10)
    s = np.sum(radii_opt)
    if s > best_sum:
        best_sum = s
        best_centers = centers_opt.copy()
        best_radii = radii_opt.copy()
    
    # Strategy 3: Corner-dominant initialization
    centers, radii = make_init_corners(n)
    result = optimize_from(centers, radii, n, max_iter=3000)
    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = np.maximum(result.x[2 * n:], 1e-10)
    s = np.sum(radii_opt)
    if s > best_sum:
        best_sum = s
        best_centers = centers_opt.copy()
        best_radii = radii_opt.copy()
    
    # Strategy 4: Random initialization (multiple seeds)
    for seed in range(5):
        np.random.seed(seed * 42 + 7)
        centers = np.random.rand(n, 2) * 0.6 + 0.2
        radii = np.ones(n) * 0.05
        result = optimize_from(centers, radii, n, max_iter=2000)
        centers_opt = result.x[:2 * n].reshape(n, 2)
        radii_opt = np.maximum(result.x[2 * n:], 1e-10)
        s = np.sum(radii_opt)
        if s > best_sum:
            best_sum = s
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    # Strategy 5: Refined grid with perturbation
    centers, radii = make_init_grid(n, spacing=0.17, radius=0.06)
    np.random.seed(123)
    centers += np.random.randn(*centers.shape) * 0.02
    centers = np.clip(centers, 0.05, 0.95)
    result = optimize_from(centers, radii, n, max_iter=3000)
    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = np.maximum(result.x[2 * n:], 1e-10)
    s = np.sum(radii_opt)
    if s > best_sum:
        best_sum = s
        best_centers = centers_opt.copy()
        best_radii = radii_opt.copy()
    
    # Force-based refinement on best result
    best_centers, best_radii = force_refine(best_centers, best_radii, n, steps=3000)
    
    # Second round of SLSQP on refined result
    result2 = optimize_from(best_centers, best_radii, n, max_iter=2000)
    centers_opt = result2.x[:2 * n].reshape(n, 2)
    radii_opt = np.maximum(result2.x[2 * n:], 1e-10)
    s2 = np.sum(radii_opt)
    if s2 > best_sum:
        best_centers = centers_opt
        best_radii = radii_opt
        best_sum = s2
    
    # Final force refinement
    best_centers, best_radii = force_refine(best_centers, best_radii, n, steps=1000)
    
    # Ensure validity
    best_radii = np.maximum(best_radii, 1e-10)
    
    return best_centers, best_radii, np.sum(best_radii)