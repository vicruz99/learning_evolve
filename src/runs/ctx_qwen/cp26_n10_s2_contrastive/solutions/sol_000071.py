# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=df5a1d17 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub = np.sqrt(dx**2 + dy**2)
    
    A_ub = np.zeros((len(I), n))
    A_ub[np.arange(len(I)), I] = 1.0
    A_ub[np.arange(len(I)), J] = 1.0
    
    # Boundary constraints via variable bounds: 0 <= r_i <= min(x, 1-x, y, 1-y)
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return res.x
    except Exception:
        pass
        
    # Fallback if LP fails
    return np.full(n, 0.01)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and squared non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + len(I))
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    c[4*N:] = dx**2 + dy**2 - (r[I] + r[J])**2
    return c

def make_hex_centers(sx, sy, shift_x, shift_y, noise_scale=0.0):
    """Generate a hexagonal lattice configuration."""
    centers = np.zeros((N, 2))
    idx = 0
    y = sy
    row = 0
    while idx < N:
        x_start = sx + (row % 2) * sx * 0.5
        col = 0
        while x_start + col * sx < 1.0 - sx and idx < N:
            centers[idx, 0] = x_start + col * sx + shift_x
            centers[idx, 1] = y + shift_y
            idx += 1
            col += 1
        y += sx * np.sqrt(3) / 2
        row += 1
        
    if idx < N:
        rng_fill = np.random.RandomState(int((shift_x + shift_y + sx) * 10000))
        for k in range(idx, N):
            centers[k] = rng_fill.uniform(0.1, 0.9, 2)
            
    if noise_scale > 0:
        rng = np.random.RandomState(int((shift_x + shift_y + sx) * 10000 + 42))
        centers += rng.normal(0, noise_scale, centers.shape)
        
    return np.clip(centers, 0.02, 0.98)

def run_packing():
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: Diverse structured starts with SLSQP + LP refinement
    starts = []
    for sx in np.linspace(0.145, 0.205, 12):
        for shift_y in np.linspace(0.01, 0.16, 6):
            starts.append((sx, shift_y, 0.0, 0.0, 0.006))
            
    for sx, sy, shx, shy, ns in starts:
        c0 = make_hex_centers(sx, sy, shx, shy, ns)
        r0 = np.full(N, 0.045)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons, options={'maxiter': 9000, 'ftol': 1e-13})
            co = np.column_stack((res.x[0::3], res.x[1::3]))
            ro = solve_lp_radii(co)
            s = np.sum(ro)
            if s > best_sum:
                best_sum = s
                best_c = co.copy()
                best_r = ro.copy()
        except Exception:
            pass

    # Phase 2: Local perturbation search on centers (LP evaluation)
    if best_c is not None:
        for trial in range(150):
            rng = np.random.RandomState(trial * 31 + 7)
            scale = 0.012 * (0.92 ** (trial // 25))
            cp = best_c + rng.normal(0, scale, best_c.shape)
            cp = np.clip(cp, 0.02, 0.98)
            rp = solve_lp_radii(cp)
            s = np.sum(rp)
            if s > best_sum:
                best_sum = s
                best_c = cp.copy()
                best_r = rp.copy()
                
        # Phase 3: Fine SLSQP refinements from perturbed LP solutions
        for trial in range(40):
            rng = np.random.RandomState(trial * 53 + 1)
            scale = 0.004
            cp = best_c + rng.normal(0, scale, best_c.shape)
            cp = np.clip(cp, 0.02, 0.98)
            rp = solve_lp_radii(cp) * 0.96
            xp = np.zeros(3 * N)
            xp[0::3] = cp[:, 0]
            xp[1::3] = cp[:, 1]
            xp[2::3] = rp
            
            try:
                res = minimize(objective, xp, method='SLSQP', bounds=bounds_opt,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
                co = np.column_stack((res.x[0::3], res.x[1::3]))
                ro = solve_lp_radii(co)
                s = np.sum(ro)
                if s > best_sum:
                    best_sum = s
                    best_c = co.copy()
                    best_r = ro.copy()
            except Exception:
                pass

    # Fallback safety net
    if best_c is None:
        best_c = make_hex_centers(0.18, 0.05, 0.0, 0.0, 0.01)
        best_r = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)

    # Strict post-processing to guarantee validity within numerical tolerance
    for i in range(N):
        mx = min(best_c[i, 0], 1.0 - best_c[i, 0], best_c[i, 1], 1.0 - best_c[i, 1])
        best_r[i] = min(best_r[i], max(0.0, mx - 1e-9))
        
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
                if d < best_r[i] + best_r[j] - 1e-9:
                    exc = best_r[i] + best_r[j] - d
                    best_r[i] -= exc / 2.0
                    best_r[j] -= exc / 2.0
                    changed = True
        if not changed:
            break
            
    best_r = np.maximum(best_r, 0.0)
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
