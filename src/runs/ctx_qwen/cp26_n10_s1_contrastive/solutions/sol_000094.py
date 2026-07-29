# sol_000094 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000079 (state c990a719) state=7fa10e6b sum of radii=2.629628 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_constraints(v):
    """
    Computes inequality constraints g(v) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    # Only upper triangular pairs to avoid duplicates and self-comparison
    i_idx, j_idx = np.triu_indices(N, k=1)
    c.append(dist2[i_idx, j_idx] - rs[i_idx, j_idx]**2)
    
    return np.concatenate(c)

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def solve_radii_lp(centers):
    """
    Given fixed centers, solves the LP to find radii that maximize sum(r_i).
    Constraints:
    1. r_i >= 0
    2. r_i <= distance to boundaries
    3. r_i + r_j <= distance(centers[i], centers[j])
    """
    n = centers.shape[0]
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        bounds_val = np.array([x, 1.0 - x, y, 1.0 - y])
        for b in bounds_val:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.full(n, 1e-5), 2.6e-4

def get_safe_radii(centers):
    """Compute strictly feasible initial radii for given centers."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        d_wall = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        d_min = np.inf
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < d_min:
                    d_min = d
        r[i] = 0.95 * min(d_wall, d_min * 0.5)
    return np.maximum(r, 1e-6)

def generate_hex_init(seed, rotation=0.0, scale=1.0):
    """Generate a hexagonal lattice initialization with specified parameters."""
    np.random.seed(seed)
    rows_counts = [6, 5, 6, 5, 4]
    pts = []
    r_est = 0.1
    y = r_est
    for r_idx, cnt in enumerate(rows_counts):
        shift = (r_idx % 2) * r_est
        x = r_est + shift
        for _ in range(cnt):
            pts.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
    pts = np.array(pts[:N])
    
    # Center and scale
    pts = pts - 0.5
    pts = pts * scale
    pts = pts + 0.5
    
    # Apply rotation around the center
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = pts @ rot_mat.T
        pts = pts - pts.mean(axis=0) + 0.5
        
    # Add controlled jitter to escape exact symmetries
    pts += np.random.randn(N, 2) * 0.015
    pts = np.clip(pts, 0.02, 0.98)
    
    return pts

def run_packing():
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Phase 1: Diverse restarts to explore global landscape
    configs = []
    np.random.seed(123)
    for i in range(40):
        rot = np.random.uniform(-0.2, 0.2)
        scale = np.random.uniform(0.9, 1.1)
        configs.append((i, rot, scale))
        
    for seed, rot, scale in configs:
        centers_init = generate_hex_init(seed, rot, scale)
        radii_init = get_safe_radii(centers_init)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = centers_init[:, 0]
        x0[1::3] = centers_init[:, 1]
        x0[2::3] = radii_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            
            if res.success:
                cons_vals = compute_constraints(res.x)
                if np.min(cons_vals) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_radii = res.x[2::3]
        except Exception:
            continue

    # Phase 2: Hill climbing with exact LP radius updates
    # This phase optimizes centers by directly maximizing the LP-derived radius sum
    if best_centers is not None:
        centers_hc = best_centers.copy()
        radii_hc, curr_sum = solve_radii_lp(centers_hc)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers_hc.copy()
            best_radii = radii_hc.copy()
            
        rng = np.random.RandomState(42)
        step_size = 0.025
        
        for iteration in range(3000):
            idx = rng.randint(N)
            old_pos = centers_hc[idx].copy()
            
            # Perturb position
            centers_hc[idx] += rng.normal(0, step_size, 2)
            centers_hc[idx] = np.clip(centers_hc[idx], 0.01, 0.99)
            
            # Solve radii exactly for new geometry
            new_radii, new_sum = solve_radii_lp(centers_hc)
            
            if new_sum > best_sum:
                best_sum = new_sum
                best_centers = centers_hc.copy()
                best_radii = new_radii.copy()
            else:
                # Revert if no improvement
                centers_hc[idx] = old_pos
                
            # Anneal step size
            step_size = max(0.001, step_size * 0.996)
            
    # Phase 3: High-precision final polish with SLSQP
    if best_centers is not None:
        x_final = np.zeros(3 * N)
        x_final[0::3] = best_centers[:, 0]
        x_final[1::3] = best_centers[:, 1]
        x_final[2::3] = best_radii
        
        try:
            res_final = minimize(objective, x_final, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                cons_vals = compute_constraints(res_final.x)
                if np.min(cons_vals) >= -1e-9:
                    final_sum = -res_final.fun
                    if final_sum > best_sum:
                        best_sum = final_sum
                        best_centers = np.column_stack((res_final.x[0::3], res_final.x[1::3]))
                        best_radii = res_final.x[2::3]
        except Exception:
            pass

    # Fallback (should not be reached given robust initialization)
    if best_centers is None:
        centers_f = np.zeros((N, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                centers_f[idx] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
                idx += 1
        centers_f[25] = [0.5, 0.5]
        r_f = np.full(N, 0.05)
        best_centers = centers_f
        best_radii = r_f
        best_sum = np.sum(r_f)

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
