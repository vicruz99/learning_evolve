import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP matrix structure
A_ub_lp = np.zeros((N_PAIRS, N))
A_ub_lp[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(N_PAIRS), J_IDX] = 1.0


def objective(x):
    return -np.sum(x[2::3])


def constraints(x):
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    c_b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_dist, c_b])


def solve_lp_radii(centers):
    n = centers.shape[0]
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-10, mx)))
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    try:
        res = linprog(-np.ones(n), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-7), 1e-6


def force_relax(centers, radii, steps=200, temp=0.01):
    c = centers.copy()
    r = radii.copy()
    n = c.shape[0]
    
    for step in range(steps):
        forces = np.zeros_like(c)
        active_temp = temp * max(0.01, 1.0 - step / steps)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = math.hypot(dx, dy)
                if d < 1e-15:
                    d = 1e-15
                    dx, dy = 1e-8, 0.0
                desired = r[i] + r[j] + 1e-6
                if d < desired:
                    f = (desired - d) / d * active_temp
                    fx, fy = dx * f, dy * f
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
        
        c += forces
        c = np.clip(c, 0.001, 0.999)
    return c


def make_pattern(rows_pattern, spacing=0.155, rng=None):
    centers = np.zeros((N, 2))
    idx = 0
    n_rows = len(rows_pattern)
    margin = 0.03
    
    if n_rows <= 0:
        return centers
    
    total_height = 1.0 - 2 * margin
    dy = total_height / (n_rows - 0.5) if n_rows > 1 else 1.0
    
    for row_i, count in enumerate(rows_pattern):
        y = margin + row_i * dy
        shift = 0.0
        if row_i % 2 == 1:
            shift = spacing * 0.5
        x_start = margin + shift
        x_end = 1.0 - margin
        if count <= 1:
            x = 0.5
        else:
            x = x_start + (x_end - x_start - (count - 1) * spacing) / 2.0 if count * spacing > x_end - x_start else x_start
        
        for _ in range(count):
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
            x += spacing
    
    while idx < N:
        if rng is None:
            rng = np.random.RandomState(idx)
        centers[idx] = rng.uniform(0.2, 0.8, 2)
        idx += 1
    
    if rng is None:
        rng = np.random.RandomState(0)
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 0.01, 0.99)


def make_hex_lattice(spacing, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    margin = spacing / 2
    y = margin
    while idx < N and y < 1.0 - margin:
        x_start = margin + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - margin and idx < N:
            centers[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        y += spacing * math.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.004, centers.shape)
    return np.clip(centers, 0.015, 0.985)


def make_strict_valid(centers, radii):
    c = centers.copy()
    r = radii.copy()
    for i in range(N):
        mx = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
        r[i] = min(r[i], max(0.0, mx - 1e-9))
    
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                if d < r[i] + r[j] - 1e-11:
                    exc = r[i] + r[j] - d
                    r[i] -= exc * 0.5
                    r[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    r = np.maximum(r, 0.0)
    return c, r


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(12345)
    
    # Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattices
    for sp in np.linspace(0.13, 0.185, 10):
        for seed in range(6):
            inits.append(make_hex_lattice(sp, seed))
    
    # Row pattern based
    patterns = [
        [7, 6, 7, 6], [6, 7, 6, 7], [8, 6, 6, 6], [6, 6, 8, 6],
        [7, 7, 6, 6], [6, 6, 7, 7], [8, 7, 6, 5], [5, 6, 7, 8],
        [7, 6, 6, 7], [6, 7, 7, 6], [9, 6, 6, 5], [5, 6, 6, 9],
        [8, 8, 5, 5], [5, 5, 8, 8], [7, 7, 7, 5], [5, 7, 7, 7],
        [6, 6, 6, 6, 2], [2, 6, 6, 6, 6], [7, 5, 7, 5, 2],
        [8, 6, 7, 5], [5, 7, 6, 8], [7, 8, 5, 6],
        [6, 6, 6, 6, 2], [4, 6, 6, 6, 4], [5, 5, 6, 5, 5],
        [5, 6, 5, 6, 4], [4, 6, 5, 6, 5], [6, 5, 6, 5, 4],
    ]
    for pat in patterns:
        for seed in range(4):
            rng_pat = np.random.RandomState(seed * 7 + 111)
            c = make_pattern(pat, spacing=0.15 + rng_pat.uniform(-0.02, 0.03), rng=rng_pat)
            inits.append(c)
    
    # Random configurations
    for seed in range(30):
        rng_r = np.random.RandomState(seed * 13 + 999)
        c = rng_r.uniform(0.1, 0.9, (N, 2))
        inits.append(c)
    
    # Phase 1: Initial optimization sweep
    for init_c in inits:
        r0, _ = solve_lp_radii(init_c)
        x0 = np.zeros(3 * N)
        x0[0::3] = init_c[:, 0]
        x0[1::3] = init_c[:, 1]
        x0[2::3] = np.maximum(r0 * 0.92, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                          constraints=cons_opt,
                          options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if -res.fun > best_sum - 1e-10:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt, s_opt = solve_lp_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping with force relaxation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(200):
            noise = 0.012 * np.exp(-step / 60.0)
            c_pert = curr_c + rng.normal(0, noise, curr_c.shape)
            c_pert = np.clip(c_pert, 0.015, 0.985)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            if s_pert > 0:
                c_relaxed = force_relax(c_pert, r_pert, steps=50, temp=0.005)
                r_rel, s_rel = solve_lp_radii(c_relaxed)
                
                accept = s_rel > curr_s or (rng.random() < 0.1 and step < 100)
                if accept:
                    x0 = np.zeros(3 * N)
                    x0[0::3] = c_relaxed[:, 0]
                    x0[1::3] = c_relaxed[:, 1]
                    x0[2::3] = np.maximum(r_rel * 0.96, 1e-5)
                    
                    try:
                        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                      constraints=cons_opt,
                                      options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                        c_new = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_new, s_new = solve_lp_radii(c_new)
                        
                        if s_new > curr_s:
                            curr_c = c_new.copy()
                            curr_r = r_new.copy()
                            curr_s = s_new
                            
                            if s_new > best_sum:
                                best_sum = s_new
                                best_centers = c_new.copy()
                                best_radii = r_new.copy()
                    except Exception:
                        pass
    
    # Phase 3: Individual circle perturbations
    if best_centers is not None:
        for iteration in range(400):
            n_pert = rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25])
            idxs = rng.choice(N, n_pert, replace=False)
            c_pert = best_centers.copy()
            scale = 0.003 + 0.005 * np.exp(-iteration / 100.0)
            c_pert[idxs] += rng.normal(0, scale, (n_pert, 2))
            c_pert = np.clip(c_pert, 0.015, 0.985)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            if s_pert > best_sum:
                x0 = np.zeros(3 * N)
                x0[0::3] = c_pert[:, 0]
                x0[1::3] = c_pert[:, 1]
                x0[2::3] = np.maximum(r_pert * 0.97, 1e-5)
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                  constraints=cons_opt,
                                  options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    c_new = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new, s_new = solve_lp_radii(c_new)
                    if s_new > best_sum:
                        best_sum = s_new
                        best_centers = c_new.copy()
                        best_radii = r_new.copy()
                except Exception:
                    pass
    
    # Phase 4: Very fine polishing
    if best_centers is not None:
        for scale in [0.001, 0.0005, 0.0002]:
            for _ in range(40):
                c_fin = best_centers + rng.normal(0, scale, best_centers.shape)
                c_fin = np.clip(c_fin, 0.01, 0.99)
                r_fin, s_fin = solve_lp_radii(c_fin)
                if s_fin > 0:
                    x0 = np.zeros(3 * N)
                    x0[0::3] = c_fin[:, 0]
                    x0[1::3] = c_fin[:, 1]
                    x0[2::3] = np.maximum(r_fin * 0.98, 1e-5)
                    try:
                        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                      constraints=cons_opt,
                                      options={'maxiter': 8000, 'ftol': 1e-15, 'disp': False})
                        c_f = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_f, s_f = solve_lp_radii(c_f)
                        if s_f > best_sum:
                            best_sum = s_f
                            best_centers = c_f.copy()
                            best_radii = r_f.copy()
                    except Exception:
                        pass
    
    # Fallback
    if best_centers is None:
        best_centers = make_hex_lattice(0.16, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # Strict post-processing
    best_centers, best_radii = make_strict_valid(best_centers, best_radii)
    
    return best_centers, best_radii, float(np.sum(best_radii))