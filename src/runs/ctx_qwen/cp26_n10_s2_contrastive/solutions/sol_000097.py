# sol_000097 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000090 (state 81009fa6) state=fca80438 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIRS_I)

def get_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    b_ub = np.zeros(NUM_PAIRS)
    
    for k in range(NUM_PAIRS):
        i, j = PAIRS_I[k], PAIRS_J[k]
        d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = d
        
    bounds = []
    for i in range(n):
        mx, my = centers[i]
        ub = min(mx, 1.0-mx, my, 1.0-my)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[PAIRS_I] - cx[PAIRS_J]
    dy = cy[PAIRS_I] - cy[PAIRS_J]
    c[4*N:] = np.hypot(dx, dy) - (r[PAIRS_I] + r[PAIRS_J])
    return c

def expand_centers(centers, radii, iters=40):
    """Push overlapping circles apart to potentially allow larger radii."""
    c = centers.copy()
    for _ in range(iters):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = PAIRS_I[k], PAIRS_J[k]
            dx = c[j,0] - c[i,0]
            dy = c[j,1] - c[i,1]
            d = np.hypot(dx, dy)
            if d < radii[i] + radii[j] - 1e-9 and d > 1e-12:
                shift = (radii[i] + radii[j] - d) * 0.505
                nx, ny = dx/d, dy/d
                c[i,0] -= nx * shift
                c[i,1] -= ny * shift
                c[j,0] += nx * shift
                c[j,1] += ny * shift
                changed = True
        if not changed:
            break
    return np.clip(c, 0.002, 0.998)

def hill_climb_centers(c_init, iters=5000, init_step=0.018):
    """Stochastic local search on center positions using SA."""
    best_c = c_init.copy()
    best_r = get_lp_radii(best_c)
    best_s = np.sum(best_r)
    
    step = init_step
    rng = np.random.default_rng(73)
    
    for i in range(iters):
        c_pert = best_c.copy()
        n_pert = rng.choice([1, 2], p=[0.75, 0.25])
        idx = rng.choice(N, n_pert, replace=False)
        c_pert[idx] += rng.normal(0, step, (n_pert, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        
        r_pert = get_lp_radii(c_pert)
        s_pert = np.sum(r_pert)
        
        if s_pert > best_s:
            best_c = c_pert
            best_r = r_pert
            best_s = s_pert
            step *= 0.96
        else:
            temp = max(0.00001, 0.04 * (1.0 - i / iters))
            if rng.random() < np.exp((s_pert - best_s) / temp):
                best_c = c_pert
                best_r = r_pert
                best_s = s_pert
                
        step *= 0.9997
        if step < 0.00003:
            step = 0.00003
            
    return best_c, best_s

def generate_inits(rng):
    """Generate diverse initial center configurations."""
    inits = []
    # Hexagonal lattices with varying densities
    for sp in np.linspace(0.18, 0.235, 6):
        c = np.zeros((N, 2))
        idx = 0
        y = sp / 2
        row = 0
        while idx < N and y < 1.0 - sp / 2:
            xs = sp / 2 + (row % 2) * sp / 2
            col = 0
            while xs + col * sp < 1.0 - sp / 2 and idx < N:
                c[idx] = [xs + col * sp, y]
                idx += 1
                col += 1
            y += sp * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.2, 0.8, 2)
            idx += 1
        c += rng.normal(0, 0.004, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # Corner and edge biased configurations
    for s in range(12):
        c = np.zeros((N, 2))
        idx = 0
        for x, y in [(0.09,0.09),(0.91,0.09),(0.09,0.91),(0.91,0.91)]:
            if idx < N:
                c[idx] = [x, y]
                idx += 1
        for v in np.linspace(0.18, 0.82, 5):
            if idx < N: c[idx] = [v, 0.09]; idx += 1
            if idx < N: c[idx] = [v, 0.91]; idx += 1
            if idx < N: c[idx] = [0.09, v]; idx += 1
            if idx < N: c[idx] = [0.91, v]; idx += 1
        while idx < N:
            c[idx] = rng.uniform(0.2, 0.8, 2)
            idx += 1
        c += rng.normal(0, 0.006, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # Pure random placements
    for _ in range(10):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    rng = np.random.default_rng(42)
    inits = generate_inits(rng)
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: SLSQP joint optimization from diverse starts
    for c0 in inits:
        r0 = get_lp_radii(c0)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = np.maximum(r0 * 0.97, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 9000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = get_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Stochastic Hill Climbing on centers
    if best_c is not None:
        c_hc, s_hc = hill_climb_centers(best_c)
        r_hc = get_lp_radii(c_hc)
        if s_hc > best_sum:
            best_sum = s_hc
            best_c = c_hc
            best_r = r_hc
            
        # Polish HC result with SLSQP
        x0 = np.zeros(3 * N)
        x0[0::3] = best_c[:, 0]
        x0[1::3] = best_c[:, 1]
        x0[2::3] = np.maximum(best_r * 0.99, 1e-5)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = get_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_c = c_opt
                    best_r = r_opt
        except Exception:
            pass
            
    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r = get_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    # Phase 3: Expand overlaps to increase inter-circle distances, then re-optimize radii
    c_exp = expand_centers(best_c, best_r)
    r_exp = get_lp_radii(c_exp)
    if np.sum(r_exp) > best_sum:
        best_sum = np.sum(r_exp)
        best_c = c_exp
        best_r = r_exp
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    for i in range(N):
        mx, my = best_c[i]
        ub = min(mx, 1.0 - mx, my, 1.0 - my)
        if best_r[i] > ub:
            best_r[i] = max(0.0, ub - 1e-9)
            
    for _ in range(40):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = PAIRS_I[k], PAIRS_J[k]
            d = np.hypot(best_c[i,0] - best_c[j,0], best_c[i,1] - best_c[j,1])
            if d < best_r[i] + best_r[j] - 1e-11:
                exc = best_r[i] + best_r[j] - d
                best_r[i] -= exc * 0.5
                best_r[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    best_r = np.maximum(best_r, 0.0)
    return best_c, best_r, float(np.sum(best_r))
