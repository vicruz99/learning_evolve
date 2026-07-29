import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute constant structure for LP pairwise constraints
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

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
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def relax_and_grow(centers, radii, iters=120, growth=0.0012, steps_per_growth=6):
    """Deterministically push circles apart while growing radii to escape overlaps."""
    c = centers.copy()
    r = radii.copy()
    r = np.maximum(r, 1e-4)
    
    for _ in range(iters):
        r *= (1.0 + growth)
        for _ in range(steps_per_growth):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = c[i, 0] - c[j, 0]
                    dy = c[i, 1] - c[j, 1]
                    d = np.hypot(dx, dy)
                    if d < 1e-9:
                        dx, dy, d = 0.001, 0.001, 0.001414
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        f = overlap * 0.5 / d
                        forces[i, 0] += dx * f
                        forces[i, 1] += dy * f
                        forces[j, 0] -= dx * f
                        forces[j, 1] -= dy * f
            
            # Boundary repulsion
            for i in range(N):
                if c[i, 0] < r[i]: forces[i, 0] += (r[i] - c[i, 0]) * 3.0
                if c[i, 0] > 1.0 - r[i]: forces[i, 0] -= (c[i, 0] - (1.0 - r[i])) * 3.0
                if c[i, 1] < r[i]: forces[i, 1] += (r[i] - c[i, 1]) * 3.0
                if c[i, 1] > 1.0 - r[i]: forces[i, 1] -= (c[i, 1] - (1.0 - r[i])) * 3.0
                
            c += forces * 0.25
            c = np.clip(c, 1e-6, 1.0 - 1e-6)
    return c, r

def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
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

def generate_hex_init(spacing, seed, shift_x=0.0, shift_y=0.0, rotation=0.0):
    """Generate hexagonal lattice initialization with controlled shifts/rotation."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = spacing / 2 + shift_y
    while idx < N and y < 1.0 - spacing / 2:
        x_start = spacing / 2 + shift_x + (row % 2) * spacing / 2
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
        
    if rotation != 0.0:
        cx, cy = centers.mean(axis=0)
        centers -= [cx, cy]
        c, s = np.cos(rotation), np.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        centers = centers @ rot.T
        centers += [cx, cy]
        
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 0.02, 0.98)

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
    
    # Phase 1: Diverse structured initializations + joint SLSQP
    inits = []
    for sp in np.linspace(0.165, 0.225, 14):
        for sx in [0.0, sp * 0.25, sp * 0.5]:
            for sy in [0.0, sp * np.sqrt(3) * 0.25]:
                inits.append(generate_hex_init(sp, seed=42, shift_x=sx, shift_y=sy))
                
    rng = np.random.RandomState(2024)
    for _ in range(25):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        if np.sum(r0) < 0.4:
            continue
            
        # Deterministic expansion to find dense packing
        c_relaxed, _ = relax_and_grow(c0, r0, iters=100, growth=0.0014)
        r_lp, s_lp = solve_lp_radii(c_relaxed)
        
        # SLSQP joint polish
        x0 = np.zeros(3 * N)
        x0[0::3] = c_relaxed[:, 0]
        x0[1::3] = c_relaxed[:, 1]
        x0[2::3] = np.maximum(r_lp * 0.94, 1e-4)
        
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
        rng = np.random.RandomState(123)
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(300):
            noise = 0.009 * np.exp(-step / 70.0)
            cp = curr_c + rng.normal(0, noise, (N, 2))
            cp = np.clip(cp, 0.02, 0.98)
            rp, sp = solve_lp_radii(cp)
            
            # Simulated annealing acceptance
            if sp > curr_s or rng.random() < np.exp((sp - curr_s) / 0.04):
                curr_c, curr_r, curr_s = cp, rp, sp
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    
                    # Local SLSQP polish after successful jump
                    x0_p = np.zeros(3 * N)
                    x0_p[0::3] = cp[:, 0]
                    x0_p[1::3] = cp[:, 1]
                    x0_p[2::3] = np.maximum(rp * 0.95, 1e-4)
                    try:
                        res_p = minimize(objective_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                                         constraints=cons_opt, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                        cx_p = res_p.x[0::3]
                        cy_p = res_p.x[1::3]
                        co_p = np.column_stack((cx_p, cy_p))
                        ro_p, so_p = solve_lp_radii(co_p)
                        if so_p > best_sum:
                            best_sum = so_p
                            best_centers = co_p.copy()
                            best_radii = ro_p.copy()
                            curr_c, curr_r, curr_s = co_p, ro_p, so_p
                    except Exception:
                        pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex_init(0.19, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 3: Strict post-processing to guarantee validator compliance
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