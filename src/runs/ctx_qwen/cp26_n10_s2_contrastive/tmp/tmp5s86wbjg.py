import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for speed
A_ub_lp = np.zeros((N_PAIRS, N))
A_ub_lp[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(N_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
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
    return np.zeros(n), 0.0

def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*N:])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    cons = np.empty(4*N + N_PAIRS)
    cons[:N] = c[:, 0] - r
    cons[N:2*N] = 1.0 - c[:, 0] - r
    cons[2*N:3*N] = c[:, 1] - r
    cons[3*N:4*N] = 1.0 - c[:, 1] - r
    
    dx = c[I_IDX, 0] - c[J_IDX, 0]
    dy = c[I_IDX, 1] - c[J_IDX, 1]
    dists = np.hypot(dx, dy)
    cons[4*N:] = dists - (r[I_IDX] + r[J_IDX])
    return cons

cons_dict = {'type': 'ineq', 'fun': constraints_joint}
bounds_opt = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N

def make_hex(spacing, rng):
    """Generate a hexagonal lattice initialization with controlled noise."""
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = spacing / 2
    while idx < N and y < 1.0 - spacing / 2:
        x = spacing / 2 + (row % 2) * spacing / 2
        while x < 1.0 - spacing / 2 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    """
    rng = np.random.RandomState(42)
    best_sum = 0.0
    best_c = None
    best_r = None

    # Phase 1: Diverse structured starts + SLSQP polish
    inits = []
    for sp in np.linspace(0.14, 0.23, 10):
        inits.append(make_hex(sp, rng))
    for _ in range(25):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))

    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        x0 = np.concatenate([c0.ravel(), np.maximum(r0 * 0.96, 1e-5)])
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_dict, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            co = res.x[:2*N].reshape(N, 2)
            ro, so = solve_lp_radii(co)
            if so > best_sum:
                best_sum = so
                best_c = co.copy()
                best_r = ro.copy()
        except Exception:
            pass

    # Phase 2: Adaptive Basin Hopping over centers
    if best_c is not None:
        cur_c = best_c.copy()
        cur_r = best_r.copy()
        cur_s = best_sum
        stagnation = 0
        
        for step in range(350):
            # Decaying noise schedule with stagnation escape
            scale = 0.018 * np.exp(-step / 60.0)
            if stagnation > 45:
                scale = 0.045
                stagnation = 0
                
            cp = best_c + rng.normal(0, scale, best_c.shape)
            cp = np.clip(cp, 0.02, 0.98)
            rp, sp = solve_lp_radii(cp)
            
            if sp > cur_s:
                cur_c, cur_r, cur_s = cp, rp, sp
                if sp > best_sum:
                    best_sum = sp
                    best_c = cp.copy()
                    best_r = rp.copy()
                    stagnation = 0
                    
                    # Local SLSQP polishing after successful jump
                    x0 = np.concatenate([cur_c.ravel(), np.maximum(cur_r * 0.98, 1e-5)])
                    try:
                        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                       constraints=cons_dict, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                        co = res.x[:2*N].reshape(N, 2)
                        ro, so = solve_lp_radii(co)
                        if so > best_sum:
                            best_sum = so
                            best_c = co.copy()
                            best_r = ro.copy()
                    except Exception:
                        pass
            else:
                stagnation += 1

    # Phase 3: Boundary & Corner Targeting
    if best_c is not None:
        for _ in range(150):
            i = rng.randint(N)
            target = rng.choice(['left', 'right', 'bottom', 'top', 'corner'])
            cp = best_c.copy()
            
            if target == 'left': cp[i, 0] = rng.uniform(0.05, 0.25)
            elif target == 'right': cp[i, 0] = rng.uniform(0.75, 0.95)
            elif target == 'bottom': cp[i, 1] = rng.uniform(0.05, 0.25)
            elif target == 'top': cp[i, 1] = rng.uniform(0.75, 0.95)
            else:
                cp[i, 0] = rng.choice([rng.uniform(0.05, 0.25), rng.uniform(0.75, 0.95)])
                cp[i, 1] = rng.choice([rng.uniform(0.05, 0.25), rng.uniform(0.75, 0.95)])
                
            cp = np.clip(cp, 0.01, 0.99)
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_c = cp.copy()
                best_r = rp.copy()

    # Phase 4: Pairwise local refinement
    if best_c is not None:
        for _ in range(80):
            i, j = rng.choice(N, 2, replace=False)
            cp = best_c.copy()
            cp[i] += rng.normal(0, 0.012, 2)
            cp[j] += rng.normal(0, 0.012, 2)
            cp = np.clip(cp, 0.01, 0.99)
            
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_c = cp.copy()
                best_r = rp.copy()
                
                x0 = np.concatenate([best_c.ravel(), np.maximum(best_r * 0.98, 1e-5)])
                try:
                    res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                    co = res.x[:2*N].reshape(N, 2)
                    ro, so = solve_lp_radii(co)
                    if so > best_sum:
                        best_sum = so
                        best_c = co.copy()
                        best_r = ro.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_c is None:
        best_c = make_hex(0.18, rng)
        best_r, best_sum = solve_lp_radii(best_c)

    # Strict post-processing to guarantee validator compliance
    c_final = best_c.copy()
    r_final = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
                if d < r_final[i] + r_final[j] - 1e-10:
                    overlap = r_final[i] + r_final[j] - d
                    r_final[i] -= overlap / 2.0
                    r_final[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))