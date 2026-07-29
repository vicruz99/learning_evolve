# sol_000127 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000111 (state 4b754d5d) state=5cca5500 sum of radii=2.568346 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
IDX_I, IDX_J = np.triu_indices(N, 1)

def compute_radii(centers):
    """Computes the maximum feasible radius for each circle given fixed centers."""
    c = np.clip(centers, 0.0, 1.0)
    xb = np.minimum(c[:, 0], 1.0 - c[:, 0])
    yb = np.minimum(c[:, 1], 1.0 - c[:, 1])
    r_bound = np.minimum(xb, yb)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair)

def objective_centers(c_flat):
    """Objective for center optimization: minimize negative sum of radii."""
    c = c_flat.reshape(N, 2)
    r = compute_radii(c)
    return -np.sum(r)

def constraints_joint(v):
    """Computes boundary and non-overlap constraints for joint optimization. Must be >= 0."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = []
    # Boundary constraints
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    # Overlap constraints (squared distance >= squared sum of radii)
    c_i = c[IDX_I]
    c_j = c[IDX_J]
    r_i = r[IDX_I]
    r_j = r[IDX_J]
    dist_sq = np.sum((c_i - c_j)**2, axis=1)
    sum_r_sq = (r_i + r_j)**2
    cons.append(dist_sq - sum_r_sq)
    return np.concatenate(cons)

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def generate_initial_configs(rng):
    """Generates diverse initial center configurations."""
    configs = []
    # 1. Hexagonal grids with varying densities
    for rows in range(4, 8):
        for cols in range(4, 8):
            if rows * cols < N: continue
            pts = []
            spacing = 0.2
            for r_idx in range(rows):
                for c_idx in range(cols):
                    x = c_idx * spacing + (spacing * 0.5 if r_idx % 2 == 1 else 0.0)
                    y = r_idx * spacing * np.sqrt(3) * 0.5
                    pts.append([x, y])
            pts = np.array(pts[:N])
            mx, my = pts.max(axis=0)
            if mx > 0: pts[:, 0] /= mx
            if my > 0: pts[:, 1] /= my
            pts = pts * 0.9 + 0.05
            configs.append(pts.copy())

    # 2. Square grids
    for side in range(5, 7):
        grid = np.linspace(0.1, 0.9, side)
        cx, cy = np.meshgrid(grid, grid)
        pts = np.column_stack((cx.flatten(), cy.flatten()))[:N]
        configs.append(pts.copy())
        
    # 3. Force-directed repulsion layouts
    for _ in range(15):
        pts = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(300):
            forces = np.zeros_like(pts)
            diff = pts[:, None, :] - pts[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1.0)
            inv_d2 = 1.0 / (dists**2 + 1e-6)
            forces += np.sum(diff * inv_d2[:, :, None], axis=1)
            pts += 0.005 * forces
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts.copy())
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    configs = generate_initial_configs(rng)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    bounds_c = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: Optimize centers with Powell (handles non-smooth radius calculation well)
    for cfg in configs:
        cfg_pert = cfg + rng.normal(0, 0.005, cfg.shape)
        cfg_pert = np.clip(cfg_pert, 0.0, 1.0)
        try:
            res = minimize(objective_centers, cfg_pert.flatten(), method='Powell', 
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-14, 'xtol': 1e-14})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = res.x.reshape(N, 2).copy()
                best_r = compute_radii(best_c)
        except Exception:
            continue
            
    # Phase 2: Refine with SLSQP on joint variables (centers + radii)
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        bounds_joint = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
        cons = {'type': 'ineq', 'fun': constraints_joint}
        
        for i in range(5):
            if i > 0:
                noise = 0.002 * (0.8 ** i)
                c_pert = best_c.copy() + rng.normal(0, noise, best_c.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                r_pert = compute_radii(c_pert) * 0.99
                v_curr = np.concatenate([c_pert.flatten(), r_pert])
            else:
                v_curr = v0.copy()
                
            try:
                res = minimize(objective_joint, v_curr, method='SLSQP',
                               bounds=bounds_joint, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False})
                if np.all(constraints_joint(res.x) >= -1e-8):
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
            except Exception:
                continue
                
    # Phase 3: Strict numerical repair to guarantee validation passes
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
