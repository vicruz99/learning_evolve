# sol_000107 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=2e78c5ac sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_constraints(vars_flat, n):
    """Computes inequality constraints: returns array where each element must be >= 0."""
    x = vars_flat[0::3]
    y = vars_flat[1::3]
    r = vars_flat[2::3]
    
    c = []
    # Boundary constraints
    c.extend(x - r)
    c.extend(1.0 - x - r)
    c.extend(y - r)
    c.extend(1.0 - y - r)
    
    # Pairwise non-overlap constraints
    i, j = np.triu_indices(n, k=1)
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dists = np.sqrt(dx**2 + dy**2)
    c.extend(dists - (r[i] + r[j]))
    
    return np.array(c)

def objective(vars_flat):
    """Minimize negative sum of radii."""
    return -np.sum(vars_flat[2::3])

def relaxation_init(n, seed):
    """Generates a feasible configuration via force-directed expansion."""
    np.random.seed(seed)
    # Hexagonal row distribution summing to 26
    rows = [5, 6, 5, 6, 4]
    r = 0.095
    centers = np.zeros((n, 2))
    idx = 0
    y = r
    for ri, cnt in enumerate(rows):
        shift = r if ri % 2 == 1 else 0.0
        x_start = r + shift
        for k in range(cnt):
            if idx < n:
                centers[idx, 0] = x_start + k * 2 * r
                centers[idx, 1] = y
                idx += 1
        y += r * np.sqrt(3)
        
    # Add small perturbation to break symmetry
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    radii = np.full(n, r)
    dt = 0.02
    
    # Iterative expansion and repulsion
    for _ in range(1000):
        radii += 0.0002
        forces = np.zeros_like(centers)
        
        # Boundary repulsion
        for d in range(2):
            viol1 = np.maximum(0, radii - centers[:, d])
            viol2 = np.maximum(0, centers[:, d] + radii - 1.0)
            forces[:, d] += (viol2 - viol1) * 200.0
            
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_d = radii[i] + radii[j]
                if dist < min_d and dist > 1e-7:
                    overlap = min_d - dist
                    dir_ = diff / dist
                    f = overlap * 100.0
                    forces[i] += dir_ * f
                    forces[j] -= dir_ * f
                    
        centers += forces * dt
        centers = np.clip(centers, 0, 1)
        
    return centers, radii

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    # Try multiple seeds to escape local minima
    for seed in range(5):
        centers, radii = relaxation_init(n, seed)
        
        x0 = np.zeros(3 * n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 2000, 'ftol': 1e-13})
            
            if res.success:
                curr_centers = res.x[:2 * n].reshape(n, 2)
                curr_radii = res.x[2 * n:]
                
                # Strict feasibility check
                c_vals = compute_constraints(res.x, n)
                if np.min(c_vals) >= -1e-5:
                    s = np.sum(curr_radii)
                    if s > best_sum:
                        best_sum = s
                        best_centers = curr_centers.copy()
                        best_radii = curr_radii.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        centers, radii = relaxation_init(n, 42)
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)
        
    # Phase 3: LP refinement for radii given fixed optimal centers
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    num_pairs = len(pairs)
    m = num_pairs + 4 * n
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i, j in pairs:
        d = np.linalg.norm(best_centers[i] - best_centers[j])
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(n):
        x, y = best_centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        lp_res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success:
            # Tiny shrink guarantees strict validity against 1e-12 tolerance
            best_radii = lp_res.x * 0.99999999
            best_sum = np.sum(best_radii)
    except Exception:
        pass
        
    # Final safety scaling loop
    for _ in range(50):
        ok = True
        for i in range(n):
            x, y = best_centers[i]
            r = best_radii[i]
            if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
                ok = False
                break
        if ok:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(best_centers[i] - best_centers[j])
                    if d < best_radii[i] + best_radii[j] - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            break
        best_radii *= 0.9999
        
    return best_centers, best_radii, float(np.sum(best_radii))
