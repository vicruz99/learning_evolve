# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=1d8dd848 sum of radii=2.590543 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_pair_indices(n):
    """Precompute indices for all unique circle pairs."""
    i_idx, j_idx = [], []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx), np.array(j_idx)

def objective_func(vars_vec, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(vars_vec[2*n:])

def constraint_func(vars_vec, n, pair_i, pair_j):
    """Compute inequality constraints: boundaries and non-overlap."""
    centers = vars_vec[:2*n].reshape(n, 2)
    radii = vars_vec[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Overlap constraints: dist >= r_i + r_j
    ci, cj = centers[pair_i], centers[pair_j]
    ri, rj = radii[pair_i], radii[pair_j]
    dist = np.sqrt(np.sum((ci - cj)**2, axis=1) + 1e-16)
    cons = np.concatenate([cons, dist - ri - rj])
    
    return cons

def hex_initialization(n, seed):
    """Generate a hexagonal lattice initialization with random perturbation."""
    np.random.seed(seed)
    r0 = 0.105
    pts = []
    y = r0
    row = 0
    while len(pts) < n + 5:
        x_start = r0 + (row % 2) * r0
        x = x_start
        while x <= 1.0 - r0:
            pts.append([x, y])
            x += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
        
    pts = np.array(pts[:n])
    # Add controlled jitter to break symmetry
    pts += np.random.uniform(-0.015, 0.015, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    # Start with feasible small radii to ensure initial validity
    radii = np.full(n, 0.04)
    return np.concatenate([pts.flatten(), radii])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = get_pair_indices(n)
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    cons_dict = {'type': 'ineq', 'fun': constraint_func, 'args': (n, pair_i, pair_j)}
    
    # Phase 1: Diverse initializations from hexagonal patterns
    for trial in range(15):
        x0 = hex_initialization(n, seed=trial * 137)
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP',
                           bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success and -res.fun > best_sum:
                c = res.x[:2*n].reshape(n, 2)
                r = res.x[2*n:]
                # Quick strict validity check
                if np.all(c[:,0] >= r - 1e-7) and np.all(c[:,0] + r <= 1.0 + 1e-7) and \
                   np.all(c[:,1] >= r - 1e-7) and np.all(c[:,1] + r <= 1.0 + 1e-7):
                    dists = np.sqrt(np.sum((c[pair_i] - c[pair_j])**2, axis=1))
                    if np.all(dists >= r[pair_i] + r[pair_j] - 1e-7):
                        s = np.sum(r)
                        if s > best_sum:
                            best_sum = s
                            best_centers = c.copy()
                            best_radii = r.copy()
        except:
            pass
            
    # Phase 2: Perturb best configuration and refine to escape local minima
    if best_centers is not None:
        for perturb_trial in range(25):
            c_pert = best_centers + np.random.uniform(-0.008, 0.008, best_centers.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            x0_pert = np.concatenate([c_pert.flatten(), best_radii])
            
            try:
                res2 = minimize(objective_func, x0_pert, args=(n,), method='SLSQP',
                                bounds=bounds, constraints=cons_dict,
                                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res2.success and -res2.fun > best_sum:
                    c2 = res2.x[:2*n].reshape(n, 2)
                    r2 = res2.x[2*n:]
                    if np.all(c2[:,0] >= r2 - 1e-7) and np.all(c2[:,0] + r2 <= 1.0 + 1e-7) and \
                       np.all(c2[:,1] >= r2 - 1e-7) and np.all(c2[:,1] + r2 <= 1.0 + 1e-7):
                        dists2 = np.sqrt(np.sum((c2[pair_i] - c2[pair_j])**2, axis=1))
                        if np.all(dists2 >= r2[pair_i] + r2[pair_j] - 1e-7):
                            s2 = np.sum(r2)
                            if s2 > best_sum:
                                best_sum = s2
                                best_centers = c2.copy()
                                best_radii = r2.copy()
            except:
                pass
                
    # Final safety adjustments to guarantee strict validity per validator rules
    centers = best_centers
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
    # Enforce non-overlap strictly with safety margin
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-9:
                    shrink = (sum_r - dist) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
