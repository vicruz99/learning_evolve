# sol_000134 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc363b95) state=2e2feba6 sum of radii=1.202496 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_config(n=26):
    """Generates a perturbed hexagonal lattice as a high-quality starting point."""
    centers = np.zeros((n, 2))
    idx = 0
    # Hexagonal row pattern: 5, 6, 5, 6, 4 = 26 circles
    rows = [5, 6, 5, 6, 4]
    y = 0.15
    dy = 0.176
    for r_idx, cnt in enumerate(rows):
        x = 0.15
        # Shift odd rows to create hexagonal packing
        if r_idx % 2 == 1:
            x = 0.25
        dx = 0.22
        for c in range(cnt):
            if idx < n:
                centers[idx] = [x + c * dx, y + r_idx * dy]
                idx += 1
        if idx == n:
            break
            
    # Add deterministic noise to break symmetry and aid optimization
    centers += np.random.default_rng(42).normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.1, 0.9)
    return centers

def run_packing():
    n = 26
    centers0 = get_initial_config(n)
    
    # Phase 1: Optimize positions for a fixed radius to relieve initial overlaps
    r_fix = 0.1
    def obj_phase1(vars):
        c = vars.reshape(n, 2)
        pen = 0.0
        for i in range(n):
            pen += max(0, r_fix - c[i,0])**2 + max(0, c[i,0] + r_fix - 1)**2
            pen += max(0, r_fix - c[i,1])**2 + max(0, c[i,1] + r_fix - 1)**2
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((c[i]-c[j])**2))
                if d < 2*r_fix:
                    pen += (2*r_fix - d)**2
        return pen
        
    res1 = minimize(obj_phase1, centers0.ravel(), method='L-BFGS-B', 
                    bounds=[(0.1, 0.9)]*(2*n), options={'maxiter': 2000})
    opt_c1 = res1.x.reshape(n, 2)
    
    # Phase 2: Jointly optimize positions and radius to maximize packing density
    def obj_phase2(vars):
        c = vars[:-1].reshape(n, 2)
        r = vars[-1]
        pen = 0.0
        wt = 2000.0
        for i in range(n):
            pen += max(0, r - c[i,0])**2 + max(0, c[i,0] + r - 1)**2
            pen += max(0, r - c[i,1])**2 + max(0, c[i,1] + r - 1)**2
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((c[i]-c[j])**2))
                if d < 2*r:
                    pen += (2*r - d)**2
        return -r + wt * pen
        
    x0_2 = np.concatenate([opt_c1.ravel(), [r_fix]])
    bounds2 = [(0.05, 0.95)]*(2*n) + [(0.01, 0.5)]
    
    res2 = minimize(obj_phase2, x0_2, method='L-BFGS-B', bounds=bounds2, 
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    final_centers = res2.x[:-1].reshape(n, 2)
    
    # Post-processing: Compute exact maximum feasible radius from optimized positions
    min_d = 1.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((final_centers[i]-final_centers[j])**2))
            if d < min_d:
                min_d = d
        min_d = min(min_d, final_centers[i,0], 1-final_centers[i,0], 
                          final_centers[i,1], 1-final_centers[i,1])
        
    # Apply safety margin to strictly satisfy validator tolerances
    r_final = min_d / 2.0 - 1e-10
    radii = np.full(n, r_final)
    return final_centers, radii, np.sum(radii)
