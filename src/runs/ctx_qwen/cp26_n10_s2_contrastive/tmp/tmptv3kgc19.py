import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(x):
    return -np.sum(x[2::3])

def constraints(x):
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4 * N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def solve_lp_radii(centers):
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    bounds = []
    for i in range(n):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-15, ub)))
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 0.001), 0.0

def make_valid(centers, radii):
    for i in range(N):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > ub:
            radii[i] = max(0.0, ub - 1e-9)
    for _ in range(200):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < radii[i] + radii[j] - 1e-12:
                exc = radii[i] + radii[j] - d
                radii[i] -= exc * 0.5
                radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def generate_inits(rng):
    inits = []
    # Hexagonal with various parameters
    for sp in np.linspace(0.13, 0.30, 25):
        for shift in [0.0, 0.02, 0.05, sp/4, sp/2]:
            c = np.zeros((N, 2))
            idx = 0
            y = 0.02
            row = 0
            while idx < N and y < 0.98:
                x = 0.02 + shift + (row % 2) * sp / 2.0
                while x < 0.98 and idx < N:
                    c[idx] = [x, y]
                    idx += 1
                    x += sp
                y += sp * math.sqrt(3) / 2.0
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            c += rng.normal(0, 0.008, c.shape)
            inits.append(np.clip(c, 0.02, 0.98))
    
    # Square grids
    for step in np.linspace(0.13, 0.27, 15):
        c = np.zeros((N, 2))
        idx = 0
        y = 0.02
        while y < 0.98 and idx < N:
            x = 0.02
            while x < 0.98 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += step
            y += step
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        inits.append(np.clip(c + rng.normal(0, 0.005, c.shape), 0.02, 0.98))
    
    # Random
    for _ in range(80):
        inits.append(np.clip(rng.uniform(0.05, 0.95, (N, 2)), 0.02, 0.98))
    
    # Corner-focused patterns
    corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
    for _ in range(20):
        c = np.zeros((N, 2))
        c[:4] = corners
        c[4:] = rng.uniform(0.1, 0.9, (N-4, 2))
        c += rng.normal(0, 0.01, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
    
    # Row-pattern init (like past solutions)
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3], [6,6,6,6,2], [5,5,5,5,6]]
    for pat in patterns:
        pts = []
        y = 0.05
        dy = 0.165
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.085
            x = 0.05 + shift
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += 0.17
            y += dy
        while len(pts) < N:
            pts.append([0.5, 0.5])
        c = np.array(pts[:N])
        c += rng.normal(0, 0.006, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
    
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start SLSQP
    for base in inits:
        c_init = np.clip(base, 0.02, 0.98)
        r_init, _ = solve_lp_radii(c_init)
        r_init = np.maximum(r_init * 0.97, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 15000, 'ftol': 1e-15, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt, _ = solve_lp_radii(c_opt)
            s_opt = np.sum(r_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
    
    if best_centers is None:
        best_centers = inits[0]
        best_radii, _ = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
    
    # Phase 2: Simulated annealing (coarse)
    curr_c = best_centers.copy()
    curr_r = best_radii.copy()
    curr_s = best_sum
    temp = 0.008
    
    for step in range(3000):
        scale = 0.010 * np.exp(-step / 1500.0)
        c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert, s_pert = solve_lp_radii(c_pert)
        
        accept = False
        if s_pert > curr_s:
            accept = True
        elif temp > 1e-10:
            delta = s_pert - curr_s
            prob = math.exp(delta / max(temp, 1e-10))
            if rng.random() < prob:
                accept = True
        
        if accept:
            curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                # SLSQP polish
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                    co = np.column_stack((res.x[0::3], res.x[1::3]))
                    ro, _ = solve_lp_radii(co)
                    so = np.sum(ro)
                    if so > best_sum:
                        best_sum = so
                        best_centers = co.copy()
                        best_radii = ro.copy()
                        curr_c, curr_r, curr_s = co, ro, so
                except Exception:
                    pass
        temp *= 0.9995
    
    # Phase 3: Greedy fine basin hopping
    curr_c = best_centers.copy()
    curr_r = best_radii.copy()
    curr_s = best_sum
    
    for step in range(12000):
        scale = 0.0015 * np.exp(-step / 5000.0)
        c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert, s_pert = solve_lp_radii(c_pert)
        if s_pert > curr_s:
            curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
    
    # Phase 4: Single circle perturbation
    for step in range(15000):
        idx = rng.integers(N)
        scale = 0.0003 * (1.0 + step / 8000.0)
        c_pert = best_centers.copy()
        c_pert[idx] += rng.normal(0, scale, 2)
        c_pert[idx] = np.clip(c_pert[idx], 0.01, 0.99)
        r_pert, s_pert = solve_lp_radii(c_pert)
        if s_pert > best_sum:
            best_sum = s_pert
            best_centers = c_pert.copy()
            best_radii = r_pert.copy()
    
    # Phase 5: Two-circle perturbation
    for step in range(8000):
        idx1, idx2 = rng.choice(N, 2, replace=False)
        scale = 0.0002
        c_pert = best_centers.copy()
        c_pert[idx1] += rng.normal(0, scale, 2)
        c_pert[idx2] += rng.normal(0, scale, 2)
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert, s_pert = solve_lp_radii(c_pert)
        if s_pert > best_sum:
            best_sum = s_pert
            best_centers = c_pert.copy()
            best_radii = r_pert.copy()
    
    # Phase 6: Restart basin hopping from best multiple times with medium noise
    for restart in range(10):
        c_start = best_centers.copy()
        c_start += rng.normal(0, 0.005, c_start.shape)
        c_start = np.clip(c_start, 0.01, 0.99)
        r_start, _ = solve_lp_radii(c_start)
        curr_c, curr_r, curr_s = c_start, r_start, np.sum(r_start)
        
        for step in range(500):
            scale = 0.003 * np.exp(-step / 200.0)
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
    
    # Phase 7: Final ultra-fine perturbation
    for step in range(8000):
        scale = 0.0001 * (1.0 + step / 4000.0)
        c_pert = best_centers.copy()
        c_pert += rng.normal(0, scale, c_pert.shape)
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert, s_pert = solve_lp_radii(c_pert)
        if s_pert > best_sum:
            best_sum = s_pert
            best_centers = c_pert.copy()
            best_radii = r_pert.copy()
    
    # Final SLSQP polish
    x0 = np.zeros(3*N)
    x0[0::3] = best_centers[:, 0]
    x0[1::3] = best_centers[:, 1]
    x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_opt,
                       options={'maxiter': 25000, 'ftol': 1e-15, 'disp': False})
        co = np.column_stack((res.x[0::3], res.x[1::3]))
        ro, _ = solve_lp_radii(co)
        so = np.sum(ro)
        if so > best_sum:
            best_sum = so
            best_centers = co.copy()
            best_radii = ro.copy()
    except Exception:
        pass
    
    # Final strict validation
    c_final, r_final = make_valid(best_centers.copy(), best_radii.copy())
    for i in range(N):
        ub = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
    
    return c_final, r_final, float(np.sum(r_final))