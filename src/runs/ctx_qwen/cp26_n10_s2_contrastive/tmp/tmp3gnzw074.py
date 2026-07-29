import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds_r.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
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

def generate_hex_init(spacing, seed, margin=0.02, rotation=0.0):
    """Generate hexagonal lattice initialization with controlled noise and rotation."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin + spacing / 2
    while idx < N and y < 1.0 - margin:
        x_start = margin + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - margin and idx < N:
            centers[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
        
    # Apply slight rotation around center to vary boundary interactions
    if abs(rotation) > 1e-5:
        c = centers - 0.5
        c = c @ np.array([[np.cos(rotation), -np.sin(rotation)], 
                           [np.sin(rotation), np.cos(rotation)]])
        centers = c + 0.5
        
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 1e-4, 1.0 - 1e-4)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    rng = np.random.RandomState(42)
    
    # Phase 1: Diverse structured initializations + joint SLSQP
    inits = []
    for sp in np.linspace(0.145, 0.235, 16):
        for s in range(6):
            for rot in [0.0, 0.05, -0.05]:
                inits.append(generate_hex_init(sp, seed=s*100+int(sp*1000), rotation=rot))
    for s in range(40):
        rng_init = np.random.RandomState(s)
        inits.append(rng_init.uniform(0.05, 0.95, (N, 2)))
        
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0, 1e-4)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.85  # Shrink to guarantee initial feasibility
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
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

    # Phase 2: LP-guided Basin Hopping with Simulated Annealing
    if best_centers is not None:
        cur_c = best_centers.copy()
        cur_r = best_radii.copy()
        cur_s = best_sum
        
        for step in range(600):
            # Adaptive noise schedule: large jumps early, fine-tune later
            noise = 0.012 * np.exp(-step / 120.0) + 0.0005
            cp = cur_c + rng.normal(0, noise, (N, 2))
            cp = np.clip(cp, 0.005, 0.995)
            
            rp, sp = solve_lp_radii(cp)
            
            # SA acceptance criterion
            accept = False
            if sp > cur_s:
                accept = True
            else:
                temp = 0.05 * np.exp(-step / 50.0)
                if rng.random() < np.exp((sp - cur_s) / max(temp, 1e-5)):
                    accept = True
                    
            if accept:
                cur_c, cur_r, cur_s = cp, rp, sp
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    
                    # Local SLSQP polish after successful jump
                    x0_p = np.zeros(3 * N)
                    x0_p[0::3] = cp[:, 0]
                    x0_p[1::3] = cp[:, 1]
                    x0_p[2::3] = rp * 0.92
                    try:
                        res_p = minimize(objective_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                                         constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                        cx_p = res_p.x[0::3]
                        cy_p = res_p.x[1::3]
                        co_p = np.column_stack((cx_p, cy_p))
                        ro_p, so_p = solve_lp_radii(co_p)
                        if so_p > best_sum:
                            best_sum = so_p
                            best_centers = co_p.copy()
                            best_radii = ro_p.copy()
                            cur_c, cur_r, cur_s = co_p, ro_p, so_p
                    except Exception:
                        pass

    # Phase 3: Targeted local search on best solution
    if best_centers is not None:
        for scale in [0.005, 0.002, 0.001]:
            for _ in range(30):
                cp = best_centers + rng.normal(0, scale, (N, 2))
                cp = np.clip(cp, 0.01, 0.99)
                rp, sp = solve_lp_radii(cp)
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    
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
    for _ in range(100):
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