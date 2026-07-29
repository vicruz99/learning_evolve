# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=0ef12d89 sum of radii=1.832036 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
NP = len(I)

def solve_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A = np.zeros((NP, n))
    A[np.arange(NP), I] = 1.0
    A[np.arange(NP), J] = 1.0
    
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b = np.sqrt(dx**2 + dy**2)
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-10):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
        
    return np.zeros(n), 0.0

def obj_slsqp(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constr_slsqp(x):
    """Inequality constraints: boundary and squared non-overlap."""
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    
    b = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    dsq = dx**2 + dy**2
    rsum = r[I] + r[J]
    o = dsq - rsum**2
    
    return np.concatenate([b, o])

def make_valid(centers, radii):
    """Strictly enforce boundary and non-overlap constraints."""
    centers = np.clip(centers, 0.0, 1.0)
    for _ in range(100):
        changed = False
        for i in range(N):
            mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mx + 1e-12:
                radii[i] = max(0.0, mx - 1e-9)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    return centers, np.maximum(radii, 0.0)

def run_packing():
    best_sum = 0.0
    best_c = None
    best_r = None
    
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constr_slsqp}
    
    # Phase 1: Diverse structured initializations
    inits = []
    for sp in np.linspace(0.13, 0.20, 10):
        c = np.zeros((N, 2))
        idx = 0
        y = sp / 2.0
        row = 0
        while idx < N and y < 1.0 - sp / 2.0:
            x = sp / 2.0 + (row % 2) * sp / 2.0
            while x < 1.0 - sp / 2.0 and idx < N:
                c[idx] = [x, y]
                x += sp
                idx += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = [0.5, 0.5]
            idx += 1
        inits.append(c)
        
    for s in range(5):
        rng = np.random.RandomState(s * 17 + 5)
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    for c0 in inits:
        r0 = np.full(N, 0.04)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                co = res.x[:2 * N].reshape(N, 2)
                ro, so = solve_lp(co)
                if so > best_sum:
                    best_sum = so
                    best_c = co.copy()
                    best_r = ro.copy()
        except Exception:
            continue

    # Phase 2: Iterative perturbation + exact LP refinement
    if best_c is not None:
        rng = np.random.RandomState(123)
        scale = 0.015
        for step in range(250):
            pert = best_c + rng.normal(0, scale, best_c.shape)
            pert = np.clip(pert, 0.01, 0.99)
            ro, so = solve_lp(pert)
            
            if so > best_sum:
                best_sum = so
                best_c = pert.copy()
                best_r = ro.copy()
                scale *= 1.05  # increase scale on success
                scale = min(scale, 0.03)
            else:
                scale *= 0.98  # decay scale on failure
                
    # Phase 3: Final SLSQP polish from best configuration
    if best_c is not None:
        x0 = np.zeros(3 * N)
        x0[0::3] = best_c[:, 0]
        x0[1::3] = best_c[:, 1]
        x0[2::3] = best_r
        
        try:
            res = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 12000, 'ftol': 1e-15, 'disp': False})
            if res.success:
                co = res.x[:2 * N].reshape(N, 2)
                ro, so = solve_lp(co)
                if so > best_sum:
                    best_sum = so
                    best_c = co.copy()
                    best_r = ro.copy()
        except Exception:
            pass

    # Fallback safety
    if best_c is None:
        best_c = inits[0].copy()
        best_r = np.full(N, 0.03)
        best_sum = np.sum(best_r)
        
    # Phase 4: Strict post-processing
    c_f, r_f = make_valid(best_c.copy(), best_r.copy())
    best_sum = float(np.sum(r_f))
    
    return c_f, r_f, best_sum
