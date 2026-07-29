# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000015 (state 9fd6082b) state=a7d8b2de sum of radii=2.601592 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import math

def get_optimal_radii(centers, n):
    """Solve LP to maximize sum of radii for fixed centers."""
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mr = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(mr, 1e-9)))
        
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    rows, cols = np.triu_indices(n, k=1)
    A_ub = np.zeros((len(rows), n))
    b_ub = dists[rows, cols]
    for k in range(len(rows)):
        A_ub[k, rows[k]] = 1.0
        A_ub[k, cols[k]] = 1.0
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        return res.x if res.success else np.full(n, 1e-5)
    except Exception:
        return np.full(n, 1e-5)

def center_penalty(params, radii, n, scale=1.0):
    """Penalty function for center optimization. Scaled radii force slack creation."""
    centers = params.reshape(n, 2)
    p = 0.0
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-12)
    rows, cols = np.triu_indices(n, k=1)
    scaled_r = radii * scale
    overlaps = (scaled_r[rows] + scaled_r[cols]) - dists[rows, cols]
    p += np.sum(np.maximum(0.0, overlaps)**2)
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r: p += (x - r)**2
        if x > 1.0 - r: p += (x + r - 1.0)**2
        if y < r: p += (y - r)**2
        if y > 1.0 - r: p += (y + r - 1.0)**2
    return p

def optimize_centers_lbfgs(centers, radii, n, scale=1.02):
    """Optimize centers using L-BFGS-B on the penalty function."""
    params0 = centers.ravel()
    bounds_c = [(0.0, 1.0)] * (2 * n)
    res = minimize(center_penalty, params0, args=(radii, n, scale), method='L-BFGS-B',
                   bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-12})
    return res.x.reshape(n, 2)

def run_packing():
    """Main packing function. Returns (centers, radii, sum_radii)."""
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice
    hex_c = []
    row = 0
    y = 0.05
    while len(hex_c) < n:
        x = 0.05 + (row % 2) * 0.0866
        while x <= 0.95 and len(hex_c) < n:
            hex_c.append([x, y])
            x += 0.1732
        y += 0.0866
        row += 1
    configs.append(np.array(hex_c[:n]))
    
    # 2. Grid layout
    grid_c = []
    for i in range(5):
        for j in range(5):
            if len(grid_c) < n:
                grid_c.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    while len(grid_c) < n:
        grid_c.append([0.5, 0.5])
    configs.append(np.array(grid_c[:n]))
    
    # 3. Random uniform
    np.random.seed(42)
    configs.append(np.random.uniform(0.1, 0.9, (n, 2)))

    # Run alternating optimization from multiple restarts
    for seed in range(6):
        np.random.seed(seed * 13 + 7)
        for cfg in configs:
            # Perturb configuration to break symmetry
            cfg_pert = cfg + np.random.randn(n, 2) * 0.02
            cfg_pert = np.clip(cfg_pert, 0.05, 0.95)
            
            centers = cfg_pert.copy()
            radii = np.full(n, 0.02)
            
            # Alternating LP / Center Optimization cycle
            for _ in range(30):
                radii = get_optimal_radii(centers, n)
                centers = optimize_centers_lbfgs(centers, radii, n, scale=1.025)
                
                # Check convergence
                pen = center_penalty(centers.ravel(), radii, n, scale=1.0)
                if pen < 1e-9:
                    break
                    
            # Final relaxation without artificial scaling
            centers = optimize_centers_lbfgs(centers, radii, n, scale=1.0)
            radii = get_optimal_radii(centers, n)
            
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # Final joint SLSQP polish to handle non-smooth transitions
    def obj_all(p):
        return -np.sum(p[2 * n:])
        
    def cons_all(p):
        c = p[:2 * n].reshape(n, 2)
        r = p[2 * n:]
        out = []
        for i in range(n):
            out.extend([c[i, 0] - r[i], 1.0 - c[i, 0] - r[i], 
                        c[i, 1] - r[i], 1.0 - c[i, 1] - r[i]])
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                out.append(d - r[i] - r[j])
        return np.array(out)
        
    bounds_all = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    x0 = np.concatenate([best_centers.ravel(), best_radii])
    try:
        res = minimize(obj_all, x0, method='SLSQP', bounds=bounds_all,
                       constraints={'type': 'ineq', 'fun': cons_all},
                       options={'maxiter': 4000, 'ftol': 1e-12})
        if res.success:
            best_centers = res.x[:2 * n].reshape(n, 2)
            best_radii = res.x[2 * n:]
            best_sum = np.sum(best_radii)
    except Exception:
        pass

    # Deterministic violation fixing to guarantee validity within tolerance
    for i in range(n):
        x, y = best_centers[i]
        mr = min(x, 1.0 - x, y, 1.0 - y)
        if best_radii[i] > mr:
            best_radii[i] = mr
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
            if best_radii[i] + best_radii[j] > d:
                exc = best_radii[i] + best_radii[j] - d
                best_radii[i] -= exc / 2.0
                best_radii[j] -= exc / 2.0
                
    best_radii = np.maximum(best_radii, 0.0)
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
