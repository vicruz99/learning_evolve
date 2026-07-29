import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for speed
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
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints for SLSQP (must be >= 0)."""
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
    """Generate hexagonal lattice initialization with controlled noise."""
    rng = np.random.RandomState(seed)
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse initializations + joint SLSQP
    inits = []
    for sp in np.linspace(0.15, 0.24, 18):
        inits.append(generate_hex_init(sp, seed=int(sp * 1000)))
    for s in range(30):
        rng_init = np.random.RandomState(s * 17 + 7)
        inits.append(rng_init.uniform(0.1, 0.9, (N, 2)))
        
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0, 1e-4)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.85  # Shrink to guarantee initial strict feasibility
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
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

    # Phase 2: Hybrid Basin Hopping on Centers with LP evaluation
    if best_centers is not None:
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        temp = 0.01
        
        for step in range(400):
            temp = 0.01 * np.exp(-step / 100.0)
            noise_scale = 0.008 * np.sqrt(max(temp, 1e-3))
            
            # Adaptive perturbation strategies to escape symmetries
            strategy = rng.choice(['perturb', 'rotate', 'swap'])
            c_pert = current_c.copy()
            
            if strategy == 'perturb':
                c_pert += rng.normal(0, noise_scale, (N, 2))
            elif strategy == 'rotate':
                angle = rng.uniform(-noise_scale, noise_scale)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                c_pert -= 0.5
                c_pert = c_pert @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                c_pert += 0.5
            elif strategy == 'swap':
                idx1, idx2 = rng.choice(N, 2, replace=False)
                c_pert[[idx1, idx2]] = c_pert[[idx2, idx1]]
                c_pert += rng.normal(0, noise_scale * 0.5, (N, 2))
                
            c_pert = np.clip(c_pert, 0.015, 0.985)
            
            rp, sp = solve_lp_radii(c_pert)
            
            # Simulated Annealing acceptance criterion
            if sp > current_s or rng.random() < np.exp((sp - current_s) / max(temp, 1e-5)):
                current_c = c_pert
                current_r = rp
                current_s = sp
                
                if sp > best_sum:
                    best_sum = sp
                    best_centers = current_c.copy()
                    best_radii = current_r.copy()
                    
                    # Local SLSQP polish after successful jump
                    x0_p = np.zeros(3 * N)
                    x0_p[0::3] = current_c[:, 0]
                    x0_p[1::3] = current_c[:, 1]
                    x0_p[2::3] = current_r * 0.95
                    try:
                        res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                         constraints=cons_opt, options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
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
                            current_s = so_p
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
    for _ in range(200):
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