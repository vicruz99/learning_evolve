# sol_000079 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state d15e4e7a) state=c990a719 sum of radii=2.631094 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

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

def get_initial_radii(pts):
    """Compute strictly feasible initial radii for given centers."""
    n = pts.shape[0]
    r = np.zeros(n)
    for i in range(n):
        max_r = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d * 0.5 < max_r:
                    max_r = d * 0.5
        r[i] = max(0.001, max_r * 0.94)
    return r

def force_simulate(n, steps, seed):
    """Spread points using force-directed layout within unit square."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)
    for _ in range(steps):
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-5)
        f_mag = 1.0 / (dists**2)
        f_mag = np.clip(f_mag, 0, 15.0)
        forces += np.sum(f_mag[:, :, None] * diff / dists[:, :, None], axis=1)
        
        # Wall repulsion
        margin = 0.05
        for d in [0, 1]:
            below = pts[:, d] < margin
            above = pts[:, d] > 1.0 - margin
            forces[below, d] += 10.0 * (margin - pts[below, d])
            forces[above, d] -= 10.0 * (pts[above, d] - (1.0 - margin))
            
        # Gentle center attraction
        forces -= 3.0 * (pts - 0.5)
        
        pts += 0.004 * forces
        pts = np.clip(pts, 0.02, 0.98)
    return pts

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
    pts += rng.uniform(-0.025, 0.025, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing():
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    inits = []
    
    # 1. Hexagonal variations
    hex_patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [4, 6, 5, 6, 5],
        [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5],
        [5, 6, 4, 6, 5],
        [4, 5, 6, 5, 6]
    ]
    for pat in hex_patterns:
        for s in range(3):
            inits.append(get_hex_init(pat, seed=s))
            
    # 2. Force simulated layouts
    for s in range(15):
        inits.append(force_simulate(N, steps=200, seed=s))
        
    # 3. Structured random rows
    for s in range(10):
        rng = np.random.RandomState(s)
        pts = rng.rand(N, 2)
        pts = pts[pts[:, 1].argsort()]
        for i in range(N):
            pts[i, 0] = 0.05 + pts[i, 0] * 0.9 + rng.uniform(-0.03, 0.03)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)

    # Run SLSQP from each initialization
    for idx, pts in enumerate(inits):
        r_init = get_initial_radii(pts)
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = pts[i, 0]
            x0[3*i+1] = pts[i, 1]
            x0[3*i+2] = r_init[i]
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2500, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = compute_constraints(res.x)
                if np.min(c_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_radii = res.x[2::3]
        except Exception:
            continue

    # Local perturbation search around the best found solution
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
                               constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = compute_constraints(res.x)
                    if np.min(c_val) >= -1e-8:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                            best_radii = res.x[2::3]
            except Exception:
                continue

    # Final high-precision polish
    if best_centers is not None:
        x_final = np.concatenate([best_centers.flatten(), best_radii])
        try:
            res_final = minimize(objective, x_final, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                c_val = compute_constraints(res_final.x)
                if np.min(c_val) >= -1e-9:
                    best_sum = -res_final.fun
                    best_centers = np.column_stack((res_final.x[0::3], res_final.x[1::3]))
                    best_radii = res_final.x[2::3]
        except Exception:
            pass

    # Fallback (should not be reached)
    if best_centers is None:
        best_centers = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                        np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        best_radii = np.full(N, 0.01)
        best_sum = np.sum(best_radii)

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
