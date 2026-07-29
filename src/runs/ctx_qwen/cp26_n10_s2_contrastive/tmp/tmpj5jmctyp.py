import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0


def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(1e-12, mx)))
    
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds_r, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return np.maximum(res.x, 0.0), -res.fun
        except Exception:
            continue
    return np.zeros(n), 0.0


def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])


def constraints_joint(x):
    """Inequality constraints (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_overlap = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    return np.concatenate([c_overlap, c_bound])


def force_relax(centers, radii, steps=100, stiffness=0.5):
    """Push overlapping circles apart using force-based relaxation."""
    c = centers.copy()
    r = radii.copy()
    n = c.shape[0]
    
    for step in range(steps):
        forces = np.zeros((n, 2))
        temp = 1.0 - step / steps
        
        for i in range(n):
            # Wall repulsion
            for wall_x in [0.0, 1.0]:
                if c[i, 0] + r[i] > wall_x - 1e-10 or c[i, 0] - r[i] < wall_x + 1e-10:
                    if wall_x == 0.0:
                        forces[i, 0] += max(0, r[i] - c[i, 0]) * 10.0
                    else:
                        forces[i, 0] -= max(0, c[i, 0] + r[i] - 1.0) * 10.0
            
            for wall_y in [0.0, 1.0]:
                if c[i, 1] + r[i] > wall_y - 1e-10 or c[i, 1] - r[i] < wall_y + 1e-10:
                    if wall_y == 0.0:
                        forces[i, 1] += max(0, r[i] - c[i, 1]) * 10.0
                    else:
                        forces[i, 1] -= max(0, c[i, 1] + r[i] - 1.0) * 10.0
            
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-12:
                    overlap = r[i] + r[j] - d
                    ux = dx / d
                    uy = dy / d
                    f = overlap * stiffness * temp
                    forces[i, 0] += ux * f
                    forces[i, 1] += uy * f
                    forces[j, 0] -= ux * f
                    forces[j, 1] -= uy * f
        
        c += forces * 0.1 * temp
        c = np.clip(c, 1e-8, 1.0 - 1e-8)
    
    return c


def generate_hex_init(spacing, seed, margin=0.04):
    """Generate hexagonal lattice with controlled spacing and noise."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = spacing / 2 + margin
    while idx < N and y < 1.0 - spacing / 2 - margin:
        x_start = spacing / 2 + margin + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - spacing / 2 - margin and idx < N:
            centers[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin + 0.05, 1.0 - margin - 0.05, 2)
        idx += 1
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)


def generate_row_pattern(pattern, seed):
    """Generate centers based on specific row patterns."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    n_rows = len(pattern)
    dy = 0.88 / (n_rows - 0.5)
    
    for r_idx, cnt in enumerate(pattern):
        y = 0.06 + r_idx * dy
        shift = 0.0 if r_idx % 2 == 0 else 0.09
        x_start = 0.06 + shift
        x_spacing = (0.88 - 2 * shift) / max(cnt - 1, 1) if cnt > 1 else 0.0
        
        for _ in range(cnt):
            if idx < N:
                centers[idx] = [x_start + np.random.uniform(-0.01, 0.01, 1)[0], y + np.random.uniform(-0.005, 0.005, 1)[0]]
                x_start += x_spacing
                idx += 1
    
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 0.02, 0.98)


def generate_corner_init(seed):
    """Generate initialization with circles favoring corners."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    
    # Place 4 circles near corners
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for i, c in enumerate(corners):
        centers[i] = c
    
    # Fill rest with perturbed hexagonal pattern
    remaining = generate_hex_init(0.18, seed)
    for i in range(4, N):
        centers[i] = remaining[i - 4]
    
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Optimizes circle packing in a unit square to maximize sum of radii."""
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattices with various spacings
    for sp in np.linspace(0.155, 0.235, 15):
        for seed in [42, 123, 456]:
            inits.append(generate_hex_init(sp, seed))
    
    # Row patterns for 26 circles
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4],
        [4, 5, 6, 5, 6], [8, 6, 5, 4, 3], [6, 6, 6, 6, 2],
        [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [7, 5, 5, 5, 4],
        [5, 7, 5, 5, 4], [4, 4, 6, 6, 6], [6, 6, 5, 5, 4],
        [5, 5, 6, 5, 5], [6, 4, 6, 5, 5], [5, 6, 4, 6, 5]
    ]
    for pat in patterns:
        for seed in range(3):
            inits.append(generate_row_pattern(pat, seed))
    
    # Corner-focused initializations
    for seed in range(10):
        inits.append(generate_corner_init(seed))
    
    # Random initializations
    rng_main = np.random.RandomState(42)
    for _ in range(30):
        inits.append(rng_main.uniform(0.08, 0.92, (N, 2)))
    
    # Phase 2: Broad search with joint SLSQP
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        if np.sum(r0) < 0.5:
            continue
        
        r0_safe = np.maximum(r0 * 0.90, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0_safe
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            co = np.column_stack((cx, cy))
            ro, so = solve_lp_radii(co)
            
            if so > best_sum:
                best_sum = so
                best_centers = co.copy()
                best_radii = ro.copy()
        except Exception:
            continue
    
    # Phase 3: LP-guided basin hopping on centers
    if best_centers is not None:
        rng = np.random.RandomState(2024)
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        
        for step in range(400):
            # Adaptive noise schedule
            noise = 0.015 * np.exp(-step / 100.0) + 0.001
            cp = current_c + rng.normal(0, noise, (N, 2))
            cp = np.clip(cp, 0.015, 0.985)
            
            rp, sp = solve_lp_radii(cp)
            
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
                current_c = cp
                current_r = rp
                
                # Polish with SLSQP
                x0_p = np.zeros(3 * N)
                x0_p[0::3] = cp[:, 0]
                x0_p[1::3] = cp[:, 1]
                x0_p[2::3] = np.maximum(rp * 0.95, 1e-5)
                try:
                    res_p = minimize(objective_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt,
                                     options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                    cx_p = res_p.x[0::3]
                    cy_p = res_p.x[1::3]
                    co_p = np.column_stack((cx_p, cy_p))
                    ro_p, so_p = solve_lp_radii(co_p)
                    if so_p > best_sum:
                        best_sum = so_p
                        best_centers = co_p.copy()
                        best_radii = ro_p.copy()
                        current_c = co_p
                        current_r = ro_p
                except Exception:
                    pass
            elif rng.random() < np.exp(min(0, (sp - best_sum) / 0.01)):
                current_c = cp
                current_r = rp
    
    # Phase 4: Individual circle perturbation
    if best_centers is not None:
        for trial in range(200):
            num_pert = rng_main.choice([1, 2, 3])
            idxs = rng_main.choice(N, num_pert, replace=False)
            
            cp = best_centers.copy()
            noise_scale = rng_main.uniform(0.002, 0.012)
            cp[idxs] += rng_main.normal(0, noise_scale, (num_pert, 2))
            cp = np.clip(cp, 0.02, 0.98)
            
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
                
                # Quick polish
                x0 = np.zeros(3 * N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.96, 1e-5)
                try:
                    res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    co = np.column_stack((cx, cy))
                    ro, so = solve_lp_radii(co)
                    if so > best_sum:
                        best_sum = so
                        best_centers = co.copy()
                        best_radii = ro.copy()
                except Exception:
                    pass
    
    # Phase 5: Force-based relaxation followed by optimization
    if best_centers is not None:
        for trial in range(50):
            cp = force_relax(best_centers, best_radii, steps=50, stiffness=0.3)
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
                
                x0 = np.zeros(3 * N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.94, 1e-5)
                try:
                    res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    co = np.column_stack((cx, cy))
                    ro, so = solve_lp_radii(co)
                    if so > best_sum:
                        best_sum = so
                        best_centers = co.copy()
                        best_radii = ro.copy()
                except Exception:
                    pass
    
    # Phase 6: Fine-grained search around best
    if best_centers is not None:
        for scale in [0.005, 0.003, 0.002, 0.001]:
            for _ in range(30):
                cp = best_centers + rng_main.normal(0, scale, (N, 2))
                cp = np.clip(cp, 0.01, 0.99)
                rp, sp = solve_lp_radii(cp)
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    
                    x0 = np.zeros(3 * N)
                    x0[0::3] = best_centers[:, 0]
                    x0[1::3] = best_centers[:, 1]
                    x0[2::3] = np.maximum(best_radii * 0.97, 1e-5)
                    try:
                        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                       constraints=cons_opt,
                                       options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
                        cx = res.x[0::3]
                        cy = res.x[1::3]
                        co = np.column_stack((cx, cy))
                        ro, so = solve_lp_radii(co)
                        if so > best_sum:
                            best_sum = so
                            best_centers = co.copy()
                            best_radii = ro.copy()
                    except Exception:
                        pass
    
    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex_init(0.19, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # Phase 7: Strict post-processing to guarantee validator compliance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], 
                 centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
    
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))