import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute LP constraint matrix for speed
A_LP = np.zeros((NUM_PAIRS, N))
A_LP[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_LP[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_slsqp(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_slsqp(x):
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

def generate_hex_init(spacing, seed):
    """Generate hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = spacing / 2
    while idx < N and y < 1.0 - spacing / 2:
        x_start = spacing / 2 + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - spacing / 2 and idx < N:
            centers[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.004, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def generate_corner_init(seed):
    """Generate initialization focused on corners and edges to maximize boundary utilization."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 2))
    # Place 4 near corners
    margin = 0.08 + rng.uniform(0, 0.04)
    centers[0] = [margin, margin]
    centers[1] = [1.0 - margin, margin]
    centers[2] = [margin, 1.0 - margin]
    centers[3] = [1.0 - margin, 1.0 - margin]
    
    # Place rest in a dense pattern in the center
    sp = 0.18 + rng.uniform(-0.02, 0.02)
    idx = 4
    y = margin + sp
    row = 0
    while idx < N and y < 1.0 - margin:
        x = margin + sp/2 + (row % 2) * sp/2
        while x < 1.0 - margin and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += sp
        y += sp * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.2, 0.8, 2)
        idx += 1
        
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_slsqp}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng_global = np.random.default_rng(42)
    
    # Phase 1: Diverse structured initializations + SLSQP
    inits = []
    for sp in np.linspace(0.155, 0.235, 12):
        inits.append(generate_hex_init(sp, seed=42))
    for s in range(8):
        inits.append(generate_corner_init(s))
    for s in range(10):
        inits.append(rng_global.uniform(0.1, 0.9, (N, 2)))
        
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0, 1e-4)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.90
        
        try:
            res = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
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

    # Phase 2: Basin Hopping / Simulated Annealing on centers
    if best_centers is not None:
        rng_sa = np.random.default_rng(123)
        current_c = best_centers.copy()
        current_r, current_sum = solve_lp_radii(current_c)
        
        sa_best_c = current_c.copy()
        sa_best_r = current_r.copy()
        sa_best_sum = current_sum
        
        T_init = 0.02
        steps = 1500
        
        for step in range(steps):
            T = T_init * np.exp(-step / 400.0)
            
            # Propose move: perturb 1-3 random circles
            num_pert = rng_sa.integers(1, 4)
            idxs = rng_sa.choice(N, num_pert, replace=False)
            noise_scale = 0.01 * np.sqrt(max(T / T_init, 0.01)) + 0.0005
            
            c_prop = current_c.copy()
            c_prop[idxs] += rng_sa.normal(0, noise_scale, (num_pert, 2))
            c_prop = np.clip(c_prop, 0.015, 0.985)
            
            r_prop, sum_prop = solve_lp_radii(c_prop)
            
            # Metropolis acceptance
            diff = sum_prop - current_sum
            accept = False
            if diff > 0:
                accept = True
            else:
                if rng_sa.random() < np.exp(diff / max(T, 1e-6)):
                    accept = True
                    
            if accept:
                current_c = c_prop
                current_r = r_prop
                current_sum = sum_prop
                
                if current_sum > sa_best_sum:
                    sa_best_sum = current_sum
                    sa_best_c = current_c.copy()
                    sa_best_r = current_r.copy()
                    
                    # Polish new SA best with short SLSQP to lock in gains
                    x0_p = np.zeros(3 * N)
                    x0_p[0::3] = sa_best_c[:, 0]
                    x0_p[1::3] = sa_best_c[:, 1]
                    x0_p[2::3] = np.maximum(sa_best_r * 0.95, 1e-5)
                    try:
                        res_p = minimize(objective_slsqp, x0_p, method='SLSQP', bounds=bounds_opt,
                                         constraints=cons_opt, options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False})
                        cx_p = res_p.x[0::3]
                        cy_p = res_p.x[1::3]
                        co_p = np.column_stack((cx_p, cy_p))
                        ro_p, so_p = solve_lp_radii(co_p)
                        if so_p > sa_best_sum:
                            sa_best_sum = so_p
                            sa_best_c = co_p.copy()
                            sa_best_r = ro_p.copy()
                    except Exception:
                        pass

        if sa_best_sum > best_sum:
            best_sum = sa_best_sum
            best_centers = sa_best_c.copy()
            best_radii = sa_best_r.copy()

    # Phase 3: Final aggressive SLSQP polish
    if best_centers is not None:
        r_final, _ = solve_lp_radii(best_centers)
        x0_final = np.zeros(3 * N)
        x0_final[0::3] = best_centers[:, 0]
        x0_final[1::3] = best_centers[:, 1]
        x0_final[2::3] = np.maximum(r_final * 0.98, 1e-5)
        
        try:
            res_final = minimize(objective_slsqp, x0_final, method='SLSQP', bounds=bounds_opt,
                                 constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            cx_f = res_final.x[0::3]
            cy_f = res_final.x[1::3]
            co_f = np.column_stack((cx_f, cy_f))
            ro_f, so_f = solve_lp_radii(co_f)
            if so_f > best_sum:
                best_sum = so_f
                best_centers = co_f.copy()
                best_radii = ro_f.copy()
        except Exception:
            pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex_init(0.19, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))