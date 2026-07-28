# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 81a0d5f4) state=8ad4a5ef sum of radii=2.463096 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_lp_radii(centers, n):
    """Solves LP to find maximum sum of radii for fixed centers."""
    c_obj = -np.ones(n)
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs + n, n))
    b_ub = np.zeros(n_pairs + n)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    for i in range(n):
        b_i = min(centers[i, 0], 1.0 - centers[i, 0],
                  centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = b_i
        idx += 1
        
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    return np.full(n, 0.05)

def objective(vars_arr, n, mu):
    """Penalty-based objective: minimize -sum(r) + penalty for violations."""
    x = vars_arr[0::3]
    y = vars_arr[1::3]
    r = vars_arr[2::3]
    
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary penalties
    pen += np.sum(np.maximum(0.0, r - x)**2)
    pen += np.sum(np.maximum(0.0, r - (1.0 - x))**2)
    pen += np.sum(np.maximum(0.0, r - y)**2)
    pen += np.sum(np.maximum(0.0, r - (1.0 - y))**2)
    
    # Overlap penalties
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, 1e6)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = np.maximum(0.0, r_sum - dist)
    pen += np.sum(np.triu(viol**2, k=1))
    
    return obj + mu * pen

def force_simulate(centers, radii, steps=1500):
    """Physics simulation to find a valid, tight packing configuration."""
    n = len(radii)
    vel = np.zeros_like(centers)
    dt = 0.005
    damping = 0.90
    k_rep = 50.0
    k_wall = 20.0
    
    for _ in range(steps):
        radii *= 1.00004
        forces = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                req = radii[i] + radii[j]
                if dist < req and dist > 1e-9:
                    overlap = req - dist
                    f = k_rep * overlap / dist
                    forces[i] += f * diff
                    forces[j] -= f * diff
                    
        for i in range(n):
            if centers[i, 0] - radii[i] < 0:
                forces[i, 0] += k_wall * (radii[i] - centers[i, 0])
            if centers[i, 0] + radii[i] > 1.0:
                forces[i, 0] -= k_wall * (centers[i, 0] + radii[i] - 1.0)
            if centers[i, 1] - radii[i] < 0:
                forces[i, 1] += k_wall * (radii[i] - centers[i, 1])
            if centers[i, 1] + radii[i] > 1.0:
                forces[i, 1] -= k_wall * (centers[i, 1] + radii[i] - 1.0)
                
        vel = damping * vel + forces * dt
        centers += vel
        centers = np.clip(centers, 0.0, 1.0)
        
    return centers, radii

def get_hex_config(n, r0):
    """Generates a hexagonal lattice configuration."""
    pts = []
    y = r0
    row = 0
    while len(pts) < n:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_res = None
    
    np.random.seed(42)
    configs = []
    
    # Diverse initial centers
    for r0 in [0.08, 0.09, 0.10]:
        cfg = get_hex_config(n, r0)
        cfg = (cfg - cfg.min(axis=0)) / (cfg.max(axis=0) - cfg.min(axis=0)) * 0.8 + 0.1
        configs.append(cfg)
        
    for _ in range(4):
        cfg = configs[0].copy()
        cfg += np.random.uniform(-0.02, 0.02, cfg.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)
        
    grid = np.array([(i*0.18+0.1, j*0.18+0.1) for j in range(5) for i in range(5)] + [[0.5, 0.5]])
    configs.append(grid[:n])
    
    for _ in range(3):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    
    for cfg in configs:
        # Phase 1: Force simulation to find feasible region
        sim_c, sim_r = force_simulate(cfg.copy(), np.full(n, 0.08))
        
        # Phase 2: LP to find tight radii for fixed centers
        lp_r = compute_lp_radii(sim_c, n)
        
        # Phase 3: Joint optimization
        x0 = np.zeros(3*n)
        x0[0::3] = sim_c[:, 0]
        x0[1::3] = sim_c[:, 1]
        x0[2::3] = lp_r * 0.995 
        
        res = minimize(objective, x0, args=(n, 2e6), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 25000, 'ftol': 1e-15})
        
        cx = res.x[0::3]
        cy = res.x[1::3]
        r = res.x[2::3]
        
        # Validation check
        valid = True
        for k in range(n):
            if cx[k] - r[k] < -1e-5 or cx[k] + r[k] > 1.0 + 1e-5 or \
               cy[k] - r[k] < -1e-5 or cy[k] + r[k] > 1.0 + 1e-5:
                valid = False; break
        if valid:
            dists = np.sqrt((cx[:,None]-cx[None,:])**2 + (cy[:,None]-cy[None,:])**2)
            np.fill_diagonal(dists, np.inf)
            r_sums = r[:,None] + r[None,:]
            if np.any(dists < r_sums - 1e-5):
                valid = False
                
        if valid:
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_res = (np.column_stack((cx, cy)), r.copy(), s)
                
    # Fallback
    if best_res is None:
        r_fb = 0.09
        c_fb = get_hex_config(n, r_fb)
        best_res = (c_fb, np.full(n, r_fb), n*r_fb)
        
    # Final safety projection
    centers, radii, _ = best_res
    scale = 1.0
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if r < 1e-12: continue
        scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i]-centers[j])
            rs = radii[i] + radii[j]
            if rs < 1e-12: continue
            scale = min(scale, d/rs)
            
    radii *= max(scale * 0.999999, 0.0)
    return centers, radii, np.sum(radii)
