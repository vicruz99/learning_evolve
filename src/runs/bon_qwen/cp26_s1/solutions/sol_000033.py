# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=01556230 sum of radii=2.576370 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(v, n):
    """Compute all inequality constraints for the packing problem.
    Returns a vector where each element must be >= 0.
    """
    m = 4 * n + n * (n - 1) // 2
    cons = np.zeros(m)
    k = 0
    for i in range(n):
        xi, yi, ri = 2 * i, 2 * i + 1, 2 * n + i
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        cons[k] = v[xi] - v[ri]
        k += 1
        cons[k] = 1.0 - v[xi] - v[ri]
        k += 1
        cons[k] = v[yi] - v[ri]
        k += 1
        cons[k] = 1.0 - v[yi] - v[ri]
        k += 1
        
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 2 * i, 2 * i + 1, 2 * n + i
            xj, yj, rj = 2 * j, 2 * j + 1, 2 * n + j
            cons[k] = (v[xi] - v[xj]) ** 2 + (v[yi] - v[yj]) ** 2 - (v[ri] + v[rj]) ** 2
            k += 1
    return cons

def force_optimize(centers, radii, steps=1500):
    """Run a force-directed simulation to resolve overlaps and expand radii."""
    n = len(radii)
    for _ in range(steps):
        forces = np.zeros_like(centers)
        max_overlap = 0.0
        
        # Inter-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = radii[i] + radii[j]
                
                if dist < min_d and dist > 1e-10:
                    overlap = min_d - dist
                    force = overlap * overlap
                    forces[i, 0] += force * dx / dist
                    forces[i, 1] += force * dy / dist
                    forces[j, 0] -= force * dx / dist
                    forces[j, 1] -= force * dy / dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                elif dist == 0.0:
                    rand_d = np.random.randn(2)
                    forces[i] += rand_d
                    forces[j] -= rand_d
                    
        # Boundary repulsion
        for i in range(n):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if x < r: forces[i, 0] += (r - x) ** 2
            if x + r > 1: forces[i, 0] -= (x + r - 1) ** 2
            if y < r: forces[i, 1] += (r - y) ** 2
            if y + r > 1: forces[i, 1] -= (y + r - 1) ** 2
            
        # Apply forces
        centers += forces * 0.1
        centers = np.clip(centers, 0.0, 1.0)
        
        # Grow radii if system is relatively stable
        if max_overlap < 1e-7:
            radii += 1e-5
            
    return centers, radii

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialize with a hexagonal lattice pattern for high initial density
    pts = []
    dy = np.sqrt(3) / 2
    for j in range(10):
        for i in range(10):
            if len(pts) >= n: break
            pts.append([i + (j % 2) * 0.5, j * dy])
        if len(pts) >= n: break
    pts = np.array(pts[:n])
    
    # Center and scale to fit comfortably inside the unit square
    pts -= pts.mean(axis=0)
    scale = 0.75 / np.max(np.abs(pts))
    pts *= scale
    pts += 0.5
    
    centers = pts
    radii = np.full(n, 0.07)
    
    # 2. Force-directed relaxation to find a dense, valid configuration
    centers, radii = force_optimize(centers, radii, steps=2000)
    
    # 3. Local optimization using SLSQP to maximize sum of radii
    x0 = np.concatenate([centers.ravel(), radii])
    
    def objective(v):
        return -np.sum(v[2 * n:])
        
    constraints = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 800, 'ftol': 1e-10, 'disp': False})
                   
    if res.success:
        centers = res.x[:2 * n].reshape(n, 2)
        radii = res.x[2 * n:]
        
    # Ensure physical validity
    radii = np.maximum(radii, 1e-9)
    centers = np.clip(centers, 0.0, 1.0)
    
    # Final safety clamp to handle numerical tolerance in constraints
    min_gap = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
            gap = d - radii[i] - radii[j]
            if gap < min_gap:
                min_gap = gap
        r = radii[i]
        gaps = [centers[i, 0] - r, 1 - centers[i, 0] - r, centers[i, 1] - r, 1 - centers[i, 1] - r]
        for g in gaps:
            if g < min_gap:
                min_gap = g
                
    if min_gap < 0:
        radii -= (np.abs(min_gap) + 1e-6)
        radii = np.maximum(radii, 0.0)
        
    return centers, radii, float(np.sum(radii))
