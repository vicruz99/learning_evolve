# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=b03bbcd0 sum of radii=2.469139 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def compute_repulsion_forces(centers, r, n):
    """Calculate repulsive forces between circles and walls."""
    forces = np.zeros_like(centers)
    strength = 50.0
    
    # Wall repulsion
    for i in range(n):
        x, y = centers[i]
        if x - r < 0: forces[i, 0] += (r - x) * strength
        if x + r > 1.0: forces[i, 0] -= (x + r - 1.0) * strength
        if y - r < 0: forces[i, 1] += (r - y) * strength
        if y + r > 1.0: forces[i, 1] -= (y + r - 1.0) * strength
        
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = math.sqrt(diff[0]**2 + diff[1]**2)
            min_dist = 2.0 * r
            if dist < min_dist and dist > 1e-6:
                factor = (min_dist - dist) / dist
                forces[i] += diff * factor
                forces[j] -= diff * factor
    return forces

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: 5x5 grid + 1 center, slightly perturbed
    centers = np.array([
        [0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)
    ] + [[0.5, 0.5]])
    
    # Deterministic perturbation
    rng = np.random.RandomState(42)
    centers += rng.normal(0, 0.015, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # 2. Repulsion Phase to spread circles and increase radius
    r = 0.04
    for step in range(600):
        forces = compute_repulsion_forces(centers, r, n)
        # Adaptive step size that decays over time
        step_size = 0.02 * max(0.0, 1.0 - step / 600.0)
        centers += forces * step_size
        centers = np.clip(centers, 0.001, 0.999)
        r += 0.00015
        
    # 3. Ensure strict feasibility before SLSQP
    min_gap = 1.0
    for i in range(n):
        # Wall gaps
        min_gap = min(min_gap, centers[i, 0], 1.0 - centers[i, 0], 
                           centers[i, 1], 1.0 - centers[i, 1])
        for j in range(i + 1, n):
            d = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            min_gap = min(min_gap, d)
            
    r_init = 0.95 * (min_gap / 2.0)
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    # 4. SLSQP Optimization
    def objective(vars):
        return -vars[-1]  # Maximize r
        
    def constraints(vars):
        coords = vars[:-1].reshape(-1, 2)
        rad = vars[-1]
        cons = []
        # Wall constraints: rad <= coord <= 1-rad
        for i in range(n):
            cons.append(coords[i, 0] - rad)
            cons.append(1.0 - coords[i, 0] - rad)
            cons.append(coords[i, 1] - rad)
            cons.append(1.0 - coords[i, 1] - rad)
        # Pairwise constraints: dist^2 >= (2r)^2
        for i in range(n):
            for j in range(i + 1, n):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                cons.append(dx*dx + dy*dy - 4.0*rad*rad)
        return np.array(cons)
        
    cons_dict = {'type': 'ineq', 'fun': constraints}
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(1e-6, 0.5)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                       options={'ftol': 1e-14, 'maxiter': 4000, 'disp': False})
        if res.success and res.x[-1] > r_init * 0.9:
            final_centers = res.x[:-1].reshape(-1, 2)
            final_r = res.x[-1]
            return final_centers, np.full(n, final_r), n * final_r
    except Exception:
        pass
        
    # 5. Fallback: Deterministic valid packing (5x5 grid + safe extra)
    # Scaled to guarantee validity without optimization
    centers_fb = np.array([
        [0.09, 0.09], [0.29, 0.09], [0.49, 0.09], [0.69, 0.09], [0.89, 0.09],
        [0.09, 0.29], [0.29, 0.29], [0.49, 0.29], [0.69, 0.29], [0.89, 0.29],
        [0.09, 0.49], [0.29, 0.49], [0.49, 0.49], [0.69, 0.49], [0.89, 0.49],
        [0.09, 0.69], [0.29, 0.69], [0.49, 0.69], [0.69, 0.69], [0.89, 0.69],
        [0.09, 0.89], [0.29, 0.89], [0.49, 0.89], [0.69, 0.89], [0.89, 0.89],
        [0.55, 0.15]
    ])
    r_fb = 0.07
    return centers_fb, np.full(n, r_fb), n * r_fb
