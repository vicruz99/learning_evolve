# sol_000259 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000245 (state 3342eff6) state=338d39d1 sum of radii=2.630829 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def obj_slqp(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def cons_slqp(v, n):
    """Inequality constraints >= 0 for valid packing."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    c = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dr = r[idx_i] + r[idx_j]
    # Use squared distance to avoid square roots in constraints
    c.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c)

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [6,6,4,6,4], 
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,6,6,7], [6,7,6,7], [5,6,5,6,5,1], [5,5,5,6,5],
        [8,5,6,7], [7,7,6,6], [9,5,6,6], [8,6,6,6]
    ]
    
    for pat in patterns:
        if sum(pat) != N_CIRCLES: continue
        pts = []
        r0 = 0.098
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx%2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N_CIRCLES: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:N_CIRCLES])
        
        # Center and scale to fit comfortably inside [0.1, 0.9]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = mx - mn
        if span[0] > 1e-4 and span[1] > 1e-4:
            pts = (pts - mn) / span * 0.75 + 0.125
        else:
            pts = np.full((N_CIRCLES, 2), 0.5)
            
        inits.append(pts)
        for _ in range(2):
            p = pts + rng.uniform(-0.015, 0.015, pts.shape)
            inits.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(6):
        inits.append(rng.uniform(0.15, 0.85, (N_CIRCLES, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_vars = [(0.0, 1.0)]*(2*N_CIRCLES) + [(1e-6, 0.5)]*N_CIRCLES
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    inits = generate_inits(rng)
    
    # Phase 1: SLSQP Multi-start Optimization
    for cfg in inits:
        x0 = np.zeros(3*N_CIRCLES)
        x0[:N_CIRCLES] = cfg[:,0]
        x0[N_CIRCLES:2*N_CIRCLES] = cfg[:,1]
        
        # Smart initial radii estimation
        r_init = np.full(N_CIRCLES, 0.09)
        for i in range(N_CIRCLES):
            dw = min(cfg[i,0], 1.0-cfg[i,0], cfg[i,1], 1.0-cfg[i,1])
            dists_i = np.sqrt(np.sum((cfg - cfg[i])**2, axis=1))
            dists_i[i] = np.inf
            dn = np.min(dists_i)
            r_init[i] = min(dw, dn/2.0) * 0.85
        x0[2*N_CIRCLES:] = r_init
        
        try:
            res = minimize(obj_slqp, x0, args=(N_CIRCLES,), method='SLSQP', bounds=bounds_vars,
                           constraints={'type': 'ineq', 'fun': cons_slqp, 'args': (N_CIRCLES,)},
                           options={'maxiter': 6000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_mat = np.column_stack((res.x[:N_CIRCLES], res.x[N_CIRCLES:2*N_CIRCLES]))
                r_lp, s_lp = solve_lp(c_mat)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_mat.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp(best_centers)

    # Phase 2: Iterative Hill-Climbing on Centers evaluated via LP
    curr_c = best_centers.copy()
    curr_r, curr_s = solve_lp(curr_c)
    step = 0.018
    
    for it in range(1000):
        idx = rng.integers(N_CIRCLES)
        old = curr_c[idx].copy()
        
        best_move = None
        best_move_s = curr_s
        
        # Try multiple random directions for this circle
        for _ in range(3):
            move = rng.uniform(-step, step, 2)
            new_c = curr_c[idx] + move
            new_c = np.clip(new_c, 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_c
            
            r_try, s_try = solve_lp(curr_c)
            if s_try > best_move_s:
                best_move_s = s_try
                best_move = new_c.copy()
                
        if best_move is not None:
            curr_c[idx] = best_move
            curr_r, curr_s = solve_lp(curr_c)
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
        else:
            curr_c[idx] = old
            
        step *= 0.9985
        
    # Phase 3: Final Joint SLSQP Polish
    x0 = np.zeros(3*N_CIRCLES)
    x0[:N_CIRCLES] = best_centers[:,0]
    x0[N_CIRCLES:2*N_CIRCLES] = best_centers[:,1]
    x0[2*N_CIRCLES:] = best_radii * 0.98
    
    try:
        res = minimize(obj_slqp, x0, args=(N_CIRCLES,), method='SLSQP', bounds=bounds_vars,
                       constraints={'type': 'ineq', 'fun': cons_slqp, 'args': (N_CIRCLES,)},
                       options={'maxiter': 10000, 'ftol': 1e-14})
        if np.isfinite(res.fun):
            c_mat = np.column_stack((res.x[:N_CIRCLES], res.x[N_CIRCLES:2*N_CIRCLES]))
            r_lp, s_lp = solve_lp(c_mat)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_mat.copy()
                best_radii = r_lp.copy()
    except Exception:
        pass

    # Phase 4: Strict Safety Scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N_CIRCLES):
        x, y, r = best_centers[i,0], best_centers[i,1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
