# sol_000075 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 25735fc7) state=7a07de3b sum of radii=2.610526 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(vars):
    """Objective function: negative sum of radii to be minimized."""
    r = vars[2 * N_CIRCLES:]
    return -np.sum(r)

def constraint_func(vars):
    """Computes inequality constraints g(x) >= 0."""
    centers = vars[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = vars[2 * N_CIRCLES:]
    
    cons = []
    
    # Boundary constraints: circles must be inside [0,1]x[0,1]
    for i in range(N_CIRCLES):
        cons.append(centers[i, 0] - radii[i])          # x - r >= 0
        cons.append(1.0 - centers[i, 0] - radii[i])    # 1 - x - r >= 0
        cons.append(centers[i, 1] - radii[i])          # y - r >= 0
        cons.append(1.0 - centers[i, 1] - radii[i])    # 1 - y - r >= 0
        
    # Non-overlap constraints: distance >= sum of radii
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            rad_sum_sq = (radii[i] + radii[j]) ** 2
            cons.append(dist_sq - rad_sum_sq)
            
    return np.array(cons)

def inflate_radii(centers, radii, max_iter=1000):
    """Conservatively increases all radii uniformly until constraints are tight."""
    delta = 1e-5
    for _ in range(max_iter):
        valid = True
        # Check boundaries
        for i in range(len(radii)):
            if radii[i] + delta > centers[i, 0] or radii[i] + delta > 1.0 - centers[i, 0] or \
               radii[i] + delta > centers[i, 1] or radii[i] + delta > 1.0 - centers[i, 1]:
                valid = False
                break
        if not valid:
            break
            
        # Check overlaps
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] + 2.0 * delta - 1e-12:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            radii = radii + delta
        else:
            break
    return radii

def run_packing():
    n = N_CIRCLES
    
    # --- Initial Configurations ---
    
    # 1. 5x5 Grid + 1 circle in a gap
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    c1 = np.array([(x, y) for x in xs for y in ys])
    c1 = np.vstack([c1, [0.2, 0.2]])
    r1 = np.full(n, 0.09)
    
    # 2. Hexagonal lattice arrangement (denser packing)
    hex_centers = []
    r_init = 0.09
    y = r_init
    # Rows: 5, 6, 5, 6, 4 circles
    for i in range(5): hex_centers.append([r_init + 2 * i * r_init, y])
    y += np.sqrt(3) * r_init
    for i in range(6): hex_centers.append([2 * r_init + 2 * i * r_init, y])
    y += np.sqrt(3) * r_init
    for i in range(5): hex_centers.append([r_init + 2 * i * r_init, y])
    y += np.sqrt(3) * r_init
    for i in range(6): hex_centers.append([2 * r_init + 2 * i * r_init, y])
    y += np.sqrt(3) * r_init
    for i in range(4): hex_centers.append([r_init + 2 * i * r_init, y])
    
    c2 = np.array(hex_centers)
    # Normalize to roughly fit in the unit square
    c2 = c2 / c2.max() * 0.9 + 0.05
    r2 = np.full(n, 0.09)
    
    # 3. Randomly perturbed grid to escape symmetric local minima
    np.random.seed(42)
    c3 = c1.copy() + np.random.uniform(-0.02, 0.02, c1.shape)
    r3 = np.full(n, 0.09)
    
    configs = [(c1, r1), (c2, r2), (c3, r3)]
    
    # Bounds for variables: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    for cent_init, rad_init in configs:
        x0 = np.concatenate([cent_init.ravel(), rad_init])
        
        result = minimize(
            objective_func,
            x0,
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            method='SLSQP',
            options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.success:
            cur_centers = result.x[:2 * n].reshape(n, 2)
            cur_radii = result.x[2 * n:]
            
            # Post-process to squeeze out remaining slack
            cur_radii = inflate_radii(cur_centers, cur_radii)
            
            cur_sum = np.sum(cur_radii)
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_centers = cur_centers
                best_radii = cur_radii
                
    # Fallback to a valid configuration if optimization fails entirely
    if best_centers is None:
        best_centers = c1
        best_radii = r1
        best_sum = np.sum(best_radii)
        
    # Ensure non-negative radii (safety)
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, best_sum
