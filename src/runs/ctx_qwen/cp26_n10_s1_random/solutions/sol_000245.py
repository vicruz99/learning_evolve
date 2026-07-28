# sol_000245 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000233 (state 4b6f20f2) state=3342eff6 sum of radii=2.629865 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        # Max radius limited by distance to nearest boundary
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def constraints(vars_arr, n):
    """Inequality constraints for SLSQP: must be >= 0."""
    x = vars_arr[:n]
    y = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    c = []
    # Boundary constraints
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap: Euclidean distance >= sum of radii
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dr = r[idx_i] + r[idx_j]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    c.append(dist - dr)
    return np.concatenate(c)

def objective(vars_arr, n):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_arr[2*n:])

def force_relax(centers, steps=400):
    """Pre-relaxation simulation to spread circles and respect boundaries."""
    c = centers.copy()
    r = np.full(len(c), 0.08)
    dt = 0.015
    
    for _ in range(steps):
        forces = np.zeros_like(c)
        
        # Wall repulsion
        for i in range(len(c)):
            if c[i,0] < r[i]: forces[i,0] += (r[i] - c[i,0]) * 20.0
            if c[i,0] > 1.0-r[i]: forces[i,0] -= (c[i,0] - (1.0-r[i])) * 20.0
            if c[i,1] < r[i]: forces[i,1] += (r[i] - c[i,1]) * 20.0
            if c[i,1] > 1.0-r[i]: forces[i,1] -= (c[i,1] - (1.0-r[i])) * 20.0
            
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        overlap = np.maximum(0.0, 2.0*r - dists)
        rep = (overlap / dists) * 60.0
        fx = np.sum(diff[:,:,0] * rep, axis=1)
        fy = np.sum(diff[:,:,1] * rep, axis=1)
        forces[:,0] += fx
        forces[:,1] += fy
        
        c += forces * dt
        c = np.clip(c, 0.01, 0.99)
        
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    bounds_vars = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    starts = []
    
    # 1. Diverse Hexagonal Patterns
    row_pats = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [6,6,4,6,4], 
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,6,6,7], [6,7,6,7], [5,6,5,6,5,1], [5,5,5,6,5]
    ]
    
    for pat in row_pats:
        if sum(pat) != n: continue
        pts = []
        r0 = 0.095
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx%2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:n])
        
        # Normalize and center configuration
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = mx - mn
        if span[0] > 1e-4 and span[1] > 1e-4:
            scale = 0.82 / max(span[0]/(1-2*r0), span[1]/(1-2*r0))
            pts = (pts - mn) * scale + 0.5 * (1.0 - scale)
        pts = np.clip(pts, 0.05, 0.95)
        starts.append(pts)
        
        # Perturbed variants
        for _ in range(3):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            starts.append(np.clip(p, 0.05, 0.95))
            
    # 2. Grid & Corner-aligned patterns
    for sp in [0.18, 0.19, 0.20]:
        grid = []
        for j in range(6):
            for i in range(5):
                if len(grid) >= n: break
                grid.append([0.1 + i*sp, 0.1 + j*sp])
        starts.append(np.array(grid[:n]))
        
    # Corner/exploitative starts
    starts.append(rng.uniform(0.15, 0.85, (n, 2)))
    for _ in range(8):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))

    # Phase 1: SLSQP Multi-start with Pre-relaxation
    for s in starts:
        s_relaxed = force_relax(s, steps=300)
        x0 = np.zeros(3*n)
        x0[:n] = s_relaxed[:,0]
        x0[n:2*n] = s_relaxed[:,1]
        
        # Smart initial radii estimation
        r_init = np.full(n, 0.085)
        for i in range(n):
            dw = min(s_relaxed[i,0], 1.0-s_relaxed[i,0], s_relaxed[i,1], 1.0-s_relaxed[i,1])
            dn = np.min(np.sqrt(np.sum((s_relaxed - s_relaxed[i])**2, axis=1) + 1e-9))
            r_init[i] = min(dw, dn / 2.0) * 0.88
        x0[2*n:] = r_init
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_vars, 
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                c_mat = np.column_stack((cx, cy))
                
                # LP refinement extracts optimal radii for these centers
                r_lp, s_lp = solve_radii_lp(c_mat)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_mat.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = starts[0]
        best_radii, best_sum = solve_radii_lp(best_centers)

    # Phase 2: Local Search on Centers evaluated via LP
    improved = True
    step_size = 0.012
    while improved and step_size > 1e-5:
        improved = False
        for _ in range(300):
            idx = rng.integers(n)
            old = best_centers[idx].copy()
            best_centers[idx] += rng.uniform(-step_size, step_size, 2)
            best_centers[idx] = np.clip(best_centers[idx], 1e-4, 1.0 - 1e-4)
            
            r_try, s_try = solve_radii_lp(best_centers)
            if s_try > best_sum + 1e-7:
                best_sum = s_try
                best_radii = r_try.copy()
                improved = True
            else:
                best_centers[idx] = old
        step_size *= 0.85
        
    # Phase 3: Final Joint SLSQP Refinement
    x0 = np.zeros(3*n)
    x0[:n] = best_centers[:,0]
    x0[n:2*n] = best_centers[:,1]
    x0[2*n:] = best_radii * 0.97
    
    try:
        res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_vars, 
                       constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13})
        if np.isfinite(res.fun):
            c_mat = np.column_stack((res.x[:n], res.x[n:2*n]))
            r_lp, s_lp = solve_radii_lp(c_mat)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_mat.copy()
                best_radii = r_lp.copy()
    except Exception:
        pass

    # Phase 4: Strict Safety Scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i,0], best_centers[i,1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
