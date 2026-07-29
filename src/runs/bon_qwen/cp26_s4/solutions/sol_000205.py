# sol_000205 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7f4d5c4f) state=4cad5097 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def hex_init(n):
    """Initialize circle centers and radii using a hexagonal lattice pattern."""
    pts = []
    r = 0.085
    for row in range(15):
        y = r + row * r * np.sqrt(3)
        if y + r > 0.95:
            break
        offset = (row % 2) * r
        for col in range(15):
            if len(pts) >= n:
                break
            x = r + offset + col * 2 * r
            if x + r > 0.95:
                break
            pts.append((x, y))
    centers = np.array(pts[:n])
    radii = np.full(n, r)
    return centers, radii

def objective(vars, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2*n:])

def constraints(vars, n):
    """Inequality constraints for boundaries and pairwise non-overlap."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    con = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    # Pairwise distance constraints: ||c_i - c_j|| >= r_i + r_j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    idx = np.triu_indices(n, k=1)
    con.append(dists[idx] - r_sum[idx])
    
    return np.concatenate(con)

def run_packing():
    n = 26
    centers, radii = hex_init(n)
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]
    
    cons = {'type': 'ineq', 'fun': lambda v: constraints(v, n)}
    
    res = minimize(
        lambda v: objective(v, n),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
    )
    
    best_c = res.x[:2*n].reshape(n, 2)
    best_r = res.x[2*n:]
    
    # Post-processing to ensure strict feasibility
    for i in range(n):
        best_r[i] = min(best_r[i], best_c[i, 0], 1-best_c[i, 0], best_c[i, 1], 1-best_c[i, 1])
        best_r[i] = max(best_r[i], 0.0)
        
    # Iteratively resolve overlaps by shrinking smaller circles
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((best_c[i] - best_c[j])**2))
                if dist < best_r[i] + best_r[j] - 1e-12:
                    if best_r[i] < best_r[j]:
                        new_r = dist - best_r[j]
                        if new_r < best_r[i]:
                            best_r[i] = max(0.0, new_r)
                            changed = True
                    else:
                        new_r = dist - best_r[i]
                        if new_r < best_r[j]:
                            best_r[j] = max(0.0, new_r)
                            changed = True
                            
    return best_c, best_r, np.sum(best_r)
