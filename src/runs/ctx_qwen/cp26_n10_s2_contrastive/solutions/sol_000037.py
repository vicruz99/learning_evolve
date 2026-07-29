# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000014 (state d34ac82b) state=3fa664a1 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP with squared-distance constraints for stability and multiple restarts for robustness.
    """
    n = 26
    
    def objective(x):
        # Minimize negative sum of radii to maximize sum
        return -np.sum(x[2::3])
        
    def constraints(x):
        cx = x[0::3]
        cy = x[1::3]
        r = x[2::3]
        
        # Preallocate constraint array
        # 4 boundary constraints per circle + C(n,2) overlap constraints
        c = np.empty(4 * n + n * (n - 1) // 2)
        idx = 0
        
        # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        c[idx:idx+n] = cx - r
        idx += n
        c[idx:idx+n] = 1.0 - cx - r
        idx += n
        c[idx:idx+n] = cy - r
        idx += n
        c[idx:idx+n] = 1.0 - cy - r
        idx += n
        
        # Overlap constraints: dx^2 + dy^2 - (ri + rj)^2 >= 0
        # Using squared distances avoids sqrt singularities and improves gradient behavior
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                c[idx] = dx*dx + dy*dy - (r[i] + r[j])**2
                idx += 1
                
        return c

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Strategy 1: Multiple restarts from perturbed hexagonal lattice
    for seed in range(25):
        np.random.seed(seed)
        
        cx = np.zeros(n)
        cy = np.zeros(n)
        r = np.full(n, 0.06)
        
        idx = 0
        row = 0
        y = 0.15 + row * 0.15 * np.sqrt(3)
        while idx < n:
            x = 0.15 + (row % 2) * 0.075
            col = 0
            while x <= 0.90 and idx < n:
                cx[idx] = x + np.random.uniform(-0.02, 0.02)
                cy[idx] = y + np.random.uniform(-0.02, 0.02)
                idx += 1
                x += 0.15
            y += 0.15 * np.sqrt(3)
            row += 1
            
        cx = np.clip(cx, 0.1, 0.9)
        cy = np.clip(cy, 0.1, 0.9)
        
        x0 = np.zeros(3 * n)
        x0[0::3] = cx
        x0[1::3] = cy
        x0[2::3] = r
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_x = res.x.copy()
        except Exception:
            continue
            
    # Strategy 2: Refine best solution with small perturbations to escape local traps
    if best_x is not None:
        for _ in range(5):
            x_pert = best_x + np.random.randn(3 * n) * 0.005
            x_pert = np.clip(x_pert, 0.0, 1.0)
            x_pert[2::3] = np.clip(x_pert[2::3], 0.001, 0.5)
            try:
                res_ref = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                if -res_ref.fun > best_sum:
                    best_sum = -res_ref.fun
                    best_x = res_ref.x.copy()
            except Exception:
                continue
                
    # Fallback if optimization failed completely
    if best_x is None:
        best_x = np.zeros(3 * n)
        best_x[0::3] = np.linspace(0.2, 0.8, n)
        best_x[1::3] = np.linspace(0.2, 0.8, n)
        best_x[2::3] = 0.05
        
    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Post-processing to guarantee strict validity within numerical tolerance
    # Enforce boundary constraints strictly
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r - 1e-9)
        
    # Iteratively resolve any remaining overlaps by proportionally shrinking radii
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((centers[i, 0]-centers[j, 0])**2 + (centers[i, 1]-centers[j, 1])**2)
                if dist < radii[i] + radii[j] - 1e-10:
                    overlap = radii[i] + radii[j] - dist
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = np.sum(radii)
    
    return centers, radii, float(final_sum)
