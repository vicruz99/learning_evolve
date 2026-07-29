# sol_000064 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05a03f22) state=2f62296e sum of radii=2.598037 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_pairwise_slack(centers, radii):
    """Computes slack for pairwise non-overlap constraints: dist - (r_i + r_j) >= 0"""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    return dists - r_sum

def get_boundary_slack(centers, radii):
    """Computes slack for boundary constraints: center +/- radius in [0, 1]"""
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    return np.stack([x - r, 1.0 - x - r, y - r, 1.0 - y - r], axis=1)

def evaluate_constraints(vars, n):
    """Combines all inequality constraints into a single vector >= 0"""
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    pw_slack = get_pairwise_slack(centers, radii)
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    pw_flat = pw_slack[mask]
    
    bnd_slack = get_boundary_slack(centers, radii)
    
    return np.concatenate([pw_flat, bnd_slack.flatten()])

def objective(vars, n):
    """Objective: maximize sum of radii (minimize negative sum)"""
    radii = vars[2*n:]
    return -np.sum(radii)

def generate_grid_init(n):
    """Generates a perturbed grid initialization"""
    # 5x5 grid is 25 circles, we need 26. 
    # We'll create a 5x6 grid subset or similar
    pts = []
    for i in range(5):
        for j in range(6):
            if len(pts) < n:
                # Slightly offset to avoid perfect symmetry issues
                x = 0.05 + j * (0.9 / 5.5)
                y = 0.05 + i * (0.9 / 4.5)
                pts.append([x, y])
    return np.array(pts[:n])

def generate_hex_init(n):
    """Generates a hexagonal pattern initialization"""
    rows_counts = [6, 5, 6, 5, 4] # Sum = 26
    pts = []
    r_est = 0.08
    dy = np.sqrt(3) * r_est
    y_curr = r_est
    
    for i, count in enumerate(rows_counts):
        dx = 2 * r_est
        width = count * dx
        x_start = (1.0 - width) / 2.0
        if i % 2 == 1:
            x_start += r_est
        for j in range(count):
            pts.append([x_start + j * dx, y_curr])
        y_curr += dy
    return np.array(pts)

def run_packing():
    n = 26
    
    # Prepare initial configurations
    inits = []
    
    # 1. Grid-like
    inits.append(generate_grid_init(n))
    
    # 2. Hexagonal-like
    inits.append(generate_hex_init(n))
    
    # 3. Random
    inits.append(np.random.rand(n, 2) * 0.8 + 0.1)
    
    # 4. Another random
    inits.append(np.random.rand(n, 2) * 0.8 + 0.1)

    best_sol = None
    best_sum = -1.0
    
    # Bounds for variables: centers in [0,1], radii >= 0
    bounds = [(0, 1)] * (2 * n) + [(0, None)] * n
    
    for c_init in inits:
        # Add slight noise to break symmetry
        c_pert = c_init + np.random.randn(*c_init.shape) * 0.02
        c_pert = np.clip(c_pert, 0.05, 0.95)
        
        # Initial radii small to ensure feasibility
        vars0 = np.concatenate([c_pert.flatten(), np.ones(n) * 0.01])
        
        # Define constraint dictionary
        cons = {'type': 'ineq', 'fun': lambda v: evaluate_constraints(v, n)}
        
        try:
            res = minimize(
                objective, 
                vars0, 
                args=(n,),
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                cur_sum = -res.fun
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_sol = res.x.copy()
        except Exception:
            pass

    if best_sol is None:
        # Fallback to simple grid
        centers = generate_grid_init(n)
        radii = np.ones(n) * 0.01
        return centers, radii, np.sum(radii)
        
    centers = best_sol[:2*n].reshape(n, 2)
    radii = best_sol[2*n:]
    
    # Ensure radii are non-negative (solver should handle it, but safe check)
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
