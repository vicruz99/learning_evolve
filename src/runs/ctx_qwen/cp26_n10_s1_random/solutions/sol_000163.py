# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000133 (state e4602328) state=5ceb6a50 sum of radii=2.626341 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective_equal(vars_arr, n):
    """Objective: maximize equal radius t -> minimize -t"""
    return -vars_arr[2 * n]

def constraints_equal(vars_arr, n):
    """Constraints for equal radius packing: boundary and pairwise separation >= 2t"""
    xs = vars_arr[:n]
    ys = vars_arr[n:2 * n]
    t = vars_arr[2 * n]
    c = np.concatenate([xs - t, 1.0 - xs - t, ys - t, 1.0 - ys - t])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    idx = np.triu_indices(n, k=1)
    c = np.concatenate([c, (dx[idx]**2 + dy[idx]**2) - 4.0 * t**2])
    return c

def objective_var(vars_arr, n):
    """Objective: maximize sum of radii -> minimize -sum(r)"""
    return -np.sum(vars_arr[2 * n:])

def constraints_var(vars_arr, n):
    """Constraints for variable radius packing: boundary and pairwise separation >= r_i + r_j"""
    xs = vars_arr[:n]
    ys = vars_arr[n:2 * n]
    rs = vars_arr[2 * n:]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    idx = np.triu_indices(n, k=1)
    c = np.concatenate([c, (dx[idx]**2 + dy[idx]**2) - dr[idx]**2])
    return c

def force_simulate(centers, n, steps=2000):
    """Physical simulation to arrange circles densely using repulsive forces"""
    r = 0.05
    dt = 0.005
    k_rep = 5.0
    k_wall = 20.0
    centers = centers.copy()
    for _ in range(steps):
        r *= 1.0003
        forces = np.zeros_like(centers)
        for i in range(n):
            for d in range(2):
                if centers[i, d] - r < 0:
                    forces[i, d] += k_wall * (r - centers[i, d])
                if centers[i, d] + r > 1.0:
                    forces[i, d] -= k_wall * (centers[i, d] + r - 1.0)
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-9
                overlap = 2.0 * r - dist
                if overlap > 0:
                    f = k_rep * overlap / dist
                    forces[i, 0] += dx * f
                    forces[i, 1] += dy * f
                    forces[j, 0] -= dx * f
                    forces[j, 1] -= dy * f
        centers += forces * dt
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
    return centers

def solve_lp_radii(centers, n):
    """Solves LP to maximize sum of radii for fixed centers"""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    idx = np.triu_indices(n, k=1)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = dists[idx]
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def compute_min_clearance(centers, n):
    """Computes the maximum equal radius feasible for the given centers"""
    xs, ys = centers[:, 0], centers[:, 1]
    d_wall = np.minimum(np.minimum(xs, 1.0 - xs), np.minimum(ys, 1.0 - ys))
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    d_pair = np.min(dists) / 2.0
    return min(np.min(d_wall), d_pair)

def generate_initial_configs(n):
    """Generates diverse high-quality initial center configurations"""
    configs = []
    row_dists = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], [4, 6, 6, 6, 4], [5, 6, 6, 5, 4]]
    for rc in row_dists:
        pts = []
        y = 0.08
        for idx, cnt in enumerate(rc):
            shift = 0.09 if idx % 2 == 1 else 0.0
            width = (cnt - 1) * 0.18
            x_start = 0.5 - width / 2.0 + shift
            for c in range(cnt):
                if len(pts) < n:
                    pts.append([x_start + c * 0.18, y])
            y += 0.155
            if len(pts) >= n: break
        configs.append(np.array(pts[:n]))
        
    rng = np.random.default_rng(123)
    for _ in range(4):
        c = rng.uniform(0.2, 0.8, (n, 2))
        configs.append(force_simulate(c, n, steps=2000))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_eq = [(0.0, 1.0)] * (2 * n) + [(0.02, 0.15)]
    bounds_var = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    
    configs = generate_initial_configs(n)
    
    for cfg in configs:
        # Dynamic initial t for equal radius optimization ensures feasibility
        clear = compute_min_clearance(cfg, n)
        t_init = max(0.05, clear * 0.7)
        
        v0 = np.concatenate([cfg.flatten(), [t_init]])
        
        try:
            # Phase 1: Tighten packing with equal radius optimization
            res_eq = minimize(objective_equal, v0, args=(n,), method='SLSQP',
                              bounds=bounds_eq, constraints={'type': 'ineq', 'fun': constraints_equal, 'args': (n,)},
                              options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            
            if np.isfinite(res_eq.fun):
                cx_eq = res_eq.x[:n]
                cy_eq = res_eq.x[n:2*n]
                centers_eq = np.column_stack((cx_eq, cy_eq))
                
                # Phase 2: LP refinement to get optimal variable radii for fixed centers
                radii_lp, _ = solve_lp_radii(centers_eq, n)
                if radii_lp is None:
                    continue
                    
                # Phase 3: Joint optimization of centers and radii
                v0_var = np.zeros(3 * n)
                v0_var[:n] = centers_eq[:, 0]
                v0_var[n:2*n] = centers_eq[:, 1]
                v0_var[2*n:] = np.maximum(radii_lp * 0.96, 1e-5)
                
                try:
                    res_var = minimize(objective_var, v0_var, args=(n,), method='SLSQP',
                                       bounds=bounds_var, constraints={'type': 'ineq', 'fun': constraints_var, 'args': (n,)},
                                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                    
                    if np.isfinite(res_var.fun):
                        cx_var = res_var.x[:n]
                        cy_var = res_var.x[n:2*n]
                        r_var = res_var.x[2*n:]
                        
                        # Strict validation before accepting
                        valid = True
                        for i in range(n):
                            if cx_var[i] < r_var[i] - 1e-9 or cx_var[i] > 1.0 - r_var[i] + 1e-9 or \
                               cy_var[i] < r_var[i] - 1e-9 or cy_var[i] > 1.0 - r_var[i] + 1e-9:
                                valid = False; break
                        if valid:
                            for i in range(n):
                                for j in range(i+1, n):
                                    d2 = (cx_var[i]-cx_var[j])**2 + (cy_var[i]-cy_var[j])**2
                                    rs = r_var[i] + r_var[j]
                                    if d2 < rs**2 - 1e-9:
                                        valid = False; break
                                if not valid: break
                        
                        if valid:
                            s = np.sum(r_var)
                            if s > best_sum:
                                best_sum = s
                                best_centers = np.column_stack((cx_var, cy_var))
                                best_radii = r_var.copy()
                except Exception:
                    pass
        except Exception:
            continue

    # Fallback configuration if optimization fails unexpectedly
    if best_centers is None:
        fallback = configs[0]
        best_centers = fallback
        radii_fb, _ = solve_lp_radii(fallback, n)
        best_radii = radii_fb if radii_fb is not None else np.full(n, 0.08)
        best_sum = np.sum(best_radii)

    # Final safety scaling to guarantee strict numerical validity for the checker
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
