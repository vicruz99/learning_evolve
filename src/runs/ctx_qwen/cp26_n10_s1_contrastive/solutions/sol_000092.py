# sol_000092 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000079 (state c990a719) state=86052361 sum of radii=2.628212 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum(r_i) subject to packing constraints."""
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
            return res.x
    except Exception:
        pass
        
    # Fallback safe radii
    return np.full(n, 0.01)

def compute_objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def compute_constraints(v):
    """Inequality constraints g(v) >= 0: boundary containment and non-overlap."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c_list = []
    # Boundary constraints
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_list.append(dist2[mask] - rs[mask]**2)
    
    return np.concatenate(c_list)

def generate_hex_init(seed, row_counts):
    """Generate hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    pts = []
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
    return np.clip(pts, 0.02, 0.98)

def generate_force_init(seed):
    """Generate force-directed layout initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    for _ in range(300):
        forces = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-8)
        f_mag = 1.0 / (dists**2 + 0.01)
        f_mag = np.clip(f_mag, 0, 10.0)
        forces += np.sum(f_mag[:, :, np.newaxis] * diff / dists[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        margin = 0.05
        for d in [0, 1]:
            below = pts[:, d] < margin
            above = pts[:, d] > 1.0 - margin
            forces[below, d] += 5.0 * (margin - pts[below, d])
            forces[above, d] -= 5.0 * (pts[above, d] - (1.0 - margin))
            
        pts += 0.005 * forces
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing():
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    inits = []
    
    # 1. Hexagonal patterns
    hex_pats = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [5, 5, 5, 5, 6], [6, 4, 6, 5, 5], [5, 6, 4, 6, 5]
    ]
    for pat in hex_pats:
        for s in range(3):
            inits.append(generate_hex_init(s, pat))
            
    # 2. Force-directed layouts
    for s in range(10):
        inits.append(generate_force_init(s))
        
    # 3. Structured grid + center
    grid_pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    grid_pts = np.vstack([grid_pts, [0.5, 0.5]])
    rng = np.random.RandomState(0)
    for _ in range(5):
        jitter = rng.uniform(-0.03, 0.03, grid_pts.shape)
        inits.append(np.clip(grid_pts + jitter, 0.02, 0.98))

    # Phase 1: SLSQP from diverse initializations
    for idx, pts in enumerate(inits):
        r_init = solve_radii_lp(pts) * 0.998  # Strict feasibility
        r_init = np.maximum(r_init, 1e-5)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = compute_constraints(res.x)
                if np.min(c_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_radii = res.x[2::3].copy()
        except Exception:
            continue

    # Phase 2: Local perturbation search to escape local minima
    if best_centers is not None:
        for _ in range(40):
            rng = np.random.RandomState(None)
            x_pert = np.concatenate([best_centers.flatten(), best_radii])
            # Perturb centers more than radii
            x_pert[0::3] += rng.normal(0, 0.005, N)
            x_pert[1::3] += rng.normal(0, 0.005, N)
            x_pert[2::3] += rng.normal(0, 0.001, N)
            
            x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
            x_pert[2::3] = np.clip(x_pert[2::3], 1e-5, 0.49)
            
            try:
                res = minimize(compute_objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = compute_constraints(res.x)
                    if np.min(c_val) >= -1e-8:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                            best_radii = res.x[2::3].copy()
            except Exception:
                continue

    # Phase 3: High-precision final polish
    if best_centers is not None:
        x_final = np.concatenate([best_centers.flatten(), best_radii])
        try:
            res_final = minimize(compute_objective, x_final, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                c_val = compute_constraints(res_final.x)
                if np.min(c_val) >= -1e-9:
                    best_sum = -res_final.fun
                    best_centers = np.column_stack((res_final.x[0::3], res_final.x[1::3]))
                    best_radii = res_final.x[2::3]
        except Exception:
            pass

    # Fallback
    if best_centers is None:
        best_centers = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                        np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        best_radii = np.full(N, 0.01)
        best_sum = np.sum(best_radii)

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
