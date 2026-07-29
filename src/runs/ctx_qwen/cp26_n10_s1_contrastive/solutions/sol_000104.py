# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000077 (state eb8dc077) state=be0dee1d sum of radii=2.633035 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def constraints(v):
    """Compute all inequality constraints: boundary containment and separation."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation constraints: dist_sq >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    c.append(dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2)
    return np.concatenate(c)

def get_radii(centers):
    """Compute strictly feasible initial radii for given centers."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d * 0.5 < max_r:
                    max_r = d * 0.5
        r[i] = max(0.001, max_r * 0.92)
    return r

def get_hex_init(row_counts, seed):
    """Generate a hexagonal lattice initialization with specified row counts."""
    pts = []
    rng = np.random.RandomState(seed)
    r_est = 0.098
    y = r_est
    for r_idx, count in enumerate(row_counts):
        x_start = r_est if r_idx % 2 == 0 else 2.0 * r_est
        for _ in range(count):
            pts.append([x_start, y])
            x_start += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
    pts = np.array(pts[:N])
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing():
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    inits = []
    
    # Diverse hexagonal patterns to capture different boundary alignments
    hex_patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [5, 5, 5, 5, 6], [6, 4, 6, 5, 5], [5, 6, 4, 6, 5],
        [4, 5, 6, 5, 6], [7, 5, 5, 5, 4], [4, 5, 5, 5, 7]
    ]
    for pat in hex_patterns:
        for s in range(4):
            inits.append(get_hex_init(pat, seed=s))
            
    # Sorted random layouts with horizontal jitter
    np.random.seed(42)
    for _ in range(15):
        pts = np.random.rand(N, 2)
        pts = pts[pts[:, 1].argsort()]
        for i in range(N):
            pts[i, 0] = 0.05 + pts[i, 0] * 0.9 + np.random.uniform(-0.03, 0.03)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)

    # Phase 1: Broad search from diverse initializations
    for pts in inits:
        r_init = get_radii(pts)
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2500, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_radii = res.x[2::3]
        except Exception:
            continue

    # Phase 2: Local perturbation refinement to escape local minima
    if best_centers is not None:
        for _ in range(25):
            rng = np.random.RandomState(None)
            x_pert = np.concatenate([best_centers.flatten(), best_radii])
            x_pert += rng.normal(0, 0.004, x_pert.shape)
            x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
            x_pert[2::3] = np.clip(x_pert[2::3], 1e-6, 0.49)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                            best_radii = res.x[2::3]
            except Exception:
                continue

        # Phase 3: High-precision polish
        try:
            x_final = np.concatenate([best_centers.flatten(), best_radii])
            res_final = minimize(objective, x_final, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                c_val = constraints(res_final.x)
                if np.min(c_val) >= -1e-7:
                    curr_sum = -res_final.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res_final.x[0::3], res_final.x[1::3]))
                        best_radii = res_final.x[2::3]
        except Exception:
            pass

    # Fallback valid configuration
    if best_centers is None:
        best_centers = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                        np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        best_radii = np.full(N, 0.01)
        best_sum = np.sum(best_radii)

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
