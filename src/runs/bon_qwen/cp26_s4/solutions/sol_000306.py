# sol_000306 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e3d19f45) state=bebf7d5a sum of radii=2.610314 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for number of circles
N_CIRCLES = 26

def compute_constraints(v, n):
    """
    Computes inequality constraints for the packing problem.
    Returns an array where all values must be >= 0.
    Constraints:
    - x >= r, 1-x >= r, y >= r, 1-y >= r (boundary)
    - dist >= r1 + r2 (non-overlap)
    """
    centers = v[:2 * n].reshape(n, 2)
    radii = v[2 * n:]
    con = []
    
    # Boundary constraints
    for i in range(n):
        con.extend([
            centers[i, 0] - radii[i],
            1.0 - centers[i, 0] - radii[i],
            centers[i, 1] - radii[i],
            1.0 - centers[i, 1] - radii[i]
        ])
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            con.append(dist - radii[i] - radii[j])
            
    return np.array(con)

def objective_function(v, n):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(v[2 * n:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate multiple initial configurations to explore the landscape
    initial_configs = []
    
    # 1. Hexagonal lattice (high density baseline)
    cx, cy = [], []
    y = 0.12
    row = 0
    while len(cx) < n:
        offset = (row % 2) * 0.15
        x = 0.12 + offset
        while x < 0.90 and len(cx) < n:
            cx.append(x)
            cy.append(y)
            x += 0.30
        y += 0.15 * np.sqrt(3)
        row += 1
    initial_configs.append(np.column_stack((cx[:n], cy[:n])))
    
    # 2. Perturbed hexagonal
    initial_configs.append(initial_configs[0] + np.random.randn(n, 2) * 0.02)
    
    # 3. Random positions
    initial_configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    # Prepare constraint and objective wrappers
    constraint_def = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    
    for init_centers in initial_configs:
        radii_init = np.full(n, 0.08)
        x0 = np.concatenate([init_centers.flatten(), radii_init])
        
        def obj_wrapper(x):
            return objective_function(x, n)
            
        try:
            res = minimize(obj_wrapper, x0, method='SLSQP', bounds=bounds,
                          constraints=constraint_def, options={'maxiter': 1500, 'ftol': 1e-10})
            
            if res.success:
                current_sum = np.sum(res.x[2 * n:])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2 * n].reshape(n, 2)
                    best_radii = res.x[2 * n:]
        except Exception:
            continue
            
    # Fallback if optimization fails completely (highly unlikely)
    if best_centers is None:
        best_centers = np.tile([0.5, 0.5], (n, 1))
        best_radii = np.full(n, 0.01)
        best_sum = np.sum(best_radii)
        
    # Final strict feasibility projection
    # Adjusts positions/radii slightly to guarantee no violations within 1e-12 tolerance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    for _ in range(20):
        needs_fix = False
        # Fix boundary violations
        for i in range(n):
            for d in range(2):
                if centers[i, d] - radii[i] < 1e-12:
                    centers[i, d] = radii[i] + 1e-12
                    needs_fix = True
                if 1.0 - centers[i, d] - radii[i] < 1e-12:
                    centers[i, d] = 1.0 - radii[i] - 1e-12
                    needs_fix = True
                    
        # Fix overlap violations by shrinking radii
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                overlap = radii[i] + radii[j] - dist
                if overlap > 1e-12:
                    # Shrink both circles equally to resolve
                    shrink = (overlap + 1e-6) / 2.0
                    radii[i] -= shrink
                    radii[j] -= shrink
                    needs_fix = True
                    
        if not needs_fix:
            break
            
    # Ensure radii are strictly positive
    radii = np.maximum(radii, 1e-8)
    
    return centers, radii, float(np.sum(radii))
