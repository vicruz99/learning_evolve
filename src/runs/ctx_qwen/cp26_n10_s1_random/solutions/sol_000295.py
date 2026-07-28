# sol_000295 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000288 (state 4522f7fa) state=cbc3e118 sum of radii=2.607929 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIU_I, TRIU_J = np.triu_indices(N_CIRCLES, k=1)

def hex_lattice_init(rows, r0):
    """Generates a hexagonal lattice configuration based on row counts."""
    pts = []
    y = r0
    for i, cnt in enumerate(rows):
        shift = r0 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= N_CIRCLES:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < N_CIRCLES:
        pts.append([0.5, 0.5])
    return np.array(pts[:N_CIRCLES])

def run_simulation(centers, steps=5000, seed=42):
    """Vectorized force-directed simulation that grows radii and resolves overlaps."""
    rng_sim = np.random.default_rng(seed)
    c = centers.copy()
    r = np.full(N_CIRCLES, 0.05)
    vel = np.zeros_like(c)
    dt = 0.004
    growth_factor = 1.00008
    
    for step in range(steps):
        # Periodic jitter to escape local minima
        if step % 400 == 0:
            c += rng_sim.uniform(-0.006, 0.006, c.shape)
            c = np.clip(c, 0.01, 0.99)
            
        r *= growth_factor
        
        # Pairwise repulsion
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dist, 1e9)
        
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0.0, r_sum - dist)
        force_mag = overlap / dist
        f_pair = np.sum(diff * force_mag[:, :, None], axis=1)
        
        # Wall repulsion
        f_wall = np.zeros_like(c)
        f_wall[:, 0] += np.clip(r - c[:, 0], 0, None) * 60.0
        f_wall[:, 0] -= np.clip(c[:, 0] + r - 1.0, 0, None) * 60.0
        f_wall[:, 1] += np.clip(r - c[:, 1], 0, None) * 60.0
        f_wall[:, 1] -= np.clip(c[:, 1] + r - 1.0, 0, None) * 60.0
        
        f = f_pair * 180.0 + f_wall
        vel = 0.82 * vel + f * dt
        c += vel
        c = np.clip(c, 1e-4, 1.0 - 1e-4)
        
    return c

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], 
                  centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    m = len(TRIU_I)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), TRIU_I] = 1.0
    A_ub[np.arange(m), TRIU_J] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[TRIU_I, TRIU_J]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2 * N_CIRCLES:])

def cons_joint(v):
    """Inequality constraints >= 0 for valid packing."""
    cx = v[:N_CIRCLES]
    cy = v[N_CIRCLES:2 * N_CIRCLES]
    r = v[2 * N_CIRCLES:]
    
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    dx = cx[TRIU_I] - cx[TRIU_J]
    dy = cy[TRIU_I] - cy[TRIU_J]
    dr = r[TRIU_I] + r[TRIU_J]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def hill_climb(centers, radii, rng, n_iter=2000):
    """Local search on centers evaluated via LP."""
    c = centers.copy()
    r, s = solve_lp(c)
    best_c, best_r, best_s = c.copy(), r.copy(), s
    
    step = 0.018
    for it in range(n_iter):
        scale = step * (1.0 - it / n_iter)
        i = rng.integers(N_CIRCLES)
        old = c[i].copy()
        c[i] += rng.uniform(-scale, scale, 2)
        c[i] = np.clip(c[i], 1e-4, 0.999)
        
        r_new, s_new = solve_lp(c)
        if s_new > best_s + 1e-9:
            best_s = s_new
            best_c = c.copy()
            best_r = r_new.copy()
        else:
            c[i] = old
            
    return best_c, best_r, best_s

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    row_patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [6,6,4,6,4],
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,6,6,7], [6,7,6,7], [5,6,5,6,5,1], [5,5,5,6,5]
    ]
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    inits = []
    for pat in row_patterns:
        if sum(pat) != N_CIRCLES:
            continue
        for r0 in [0.085, 0.095, 0.105]:
            pts = hex_lattice_init(pat, r0)
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            span = mx - mn
            if np.any(span > 1e-4):
                pts = (pts - mn) / span * 0.75 + 0.125
            inits.append(pts)
            
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (N_CIRCLES, 2)))
        
    # Phase 1: Simulation + LP
    for cfg in inits:
        c_sim = run_simulation(cfg)
        r_lp, s_lp = solve_lp(c_sim)
        if s_lp > best_sum:
            best_sum = s_lp
            best_centers = c_sim.copy()
            best_radii = r_lp.copy()
            
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp(best_centers)
        
    # Phase 2: Joint SLSQP Polish
    x0 = np.zeros(3 * N_CIRCLES)
    x0[:N_CIRCLES] = best_centers[:, 0]
    x0[N_CIRCLES:2 * N_CIRCLES] = best_centers[:, 1]
    x0[2 * N_CIRCLES:] = best_radii * 0.97
    
    bounds_opt = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(1e-6, 0.5)] * N_CIRCLES
    try:
        res = minimize(obj_joint, x0, method='SLSQP', bounds=bounds_opt,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 10000, 'ftol': 1e-14})
        if np.isfinite(res.fun):
            c_opt = np.column_stack((res.x[:N_CIRCLES], res.x[N_CIRCLES:2 * N_CIRCLES]))
            r_opt, s_opt = solve_lp(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
    except Exception:
        pass
        
    # Phase 3: Hill Climbing on Centers
    best_centers, best_radii, best_sum = hill_climb(best_centers, best_radii, rng)
    
    # Phase 4: Strict Safety Scaling
    scale = 1.0
    for i in range(N_CIRCLES):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            d = dists[i, j]
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
