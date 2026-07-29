import numpy as np
from scipy.optimize import minimize, linprog
import math
import warnings

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute sparse LP constraint matrix structure
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    c_obj = -np.ones(N)
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds_r.append((0.0, max(1e-9, mx)))
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
        
    # Fallback solver
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds_r, method='interior-point')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
        
    return np.zeros(N), 0.0

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

def generate_initializations(rng):
    """Generate diverse starting configurations."""
    inits = []
    
    # 1. Hexagonal lattices with varying densities and vertical shifts
    for sp in np.linspace(0.155, 0.245, 16):
        for shift_y in [0.0, 0.03, 0.07]:
            c = np.zeros((N, 2))
            idx = 0
            row = 0
            y = sp / 2 + shift_y
            while idx < N and y < 1.0 - sp / 2:
                x_start = sp / 2 + (row % 2) * sp / 2
                col = 0
                while x_start + col * sp < 1.0 - sp / 2 and idx < N:
                    c[idx] = [x_start + col * sp, y]
                    idx += 1
                    col += 1
                y += sp * math.sqrt(3) / 2
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            inits.append(c + rng.normal(0, 0.004, c.shape))
            
    # 2. Random dense packings
    for _ in range(35):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # 3. Boundary/Corner pinned configurations
    for seed in range(15):
        c = rng.uniform(0.06, 0.94, (N, 2))
        # Explicitly place circles near corners to exploit boundary slack
        c[0] = [0.09, 0.09]
        c[1] = [0.91, 0.09]
        c[2] = [0.09, 0.91]
        c[3] = [0.91, 0.91]
        inits.append(c + rng.normal(0, 0.002, c.shape))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_initializations(rng)
    
    # Phase 1: Broad SLSQP search from structured starts
    for c0 in inits:
        c0 = np.clip(c0, 0.02, 0.98)
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0 * 0.85, 1e-4)  # Shrink ensures strict feasibility
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
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

    # Phase 2: Simulated Annealing on Centers guided by LP radius sums
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_s = best_sum
        
        for step in range(1000):
            # Exponential cooling
            temp = 0.025 * np.exp(-step / 120.0)
            noise_scale = 0.008 * np.sqrt(max(temp, 1e-4))
            
            # Perturb a random subset of circles
            num_pert = rng.integers(1, 7)
            idxs = rng.choice(N, num_pert, replace=False)
            c_pert = curr_c.copy()
            c_pert[idxs] += rng.normal(0, noise_scale, (num_pert, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            # Simulated annealing acceptance
            delta = s_pert - curr_s
            if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-4)):
                curr_c = c_pert
                curr_s = s_pert
                
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_centers = curr_c.copy()
                    best_radii = r_pert.copy()
                    
                    # Local SLSQP polish on new global best
                    x0_p = np.zeros(3 * N)
                    x0_p[0::3] = curr_c[:, 0]
                    x0_p[1::3] = curr_c[:, 1]
                    x0_p[2::3] = best_radii * 0.92
                    try:
                        res_p = minimize(objective_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                                         constraints=cons_opt,
                                         options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                        cx_p = res_p.x[0::3]
                        cy_p = res_p.x[1::3]
                        co_p = np.column_stack((cx_p, cy_p))
                        ro_p, so_p = solve_lp_radii(co_p)
                        if so_p > best_sum:
                            best_sum = so_p
                            best_centers = co_p.copy()
                            best_radii = ro_p.copy()
                            curr_c = co_p
                    except Exception:
                        pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_initializations(np.random.default_rng(0))[0]
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
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = math.hypot(dx, dy)
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))