# sol_000246 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000233 (state 4b6f20f2) state=df3fca82 sum of radii=2.620157 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def slesqp_objective(v, n):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(v[2*n:])

def slesqp_constraints(v, n):
    """Inequality constraints for SLSQP: must be >= 0."""
    cx, cy, r = v[:n], v[n:2*n], v[2*n:]
    c = []
    # Boundary constraints
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    # Pairwise non-overlap using Euclidean distance for better gradients
    idx = np.triu_indices(n, 1)
    dx = cx[idx[0]] - cx[idx[1]]
    dy = cy[idx[0]] - cy[idx[1]]
    c.append(np.sqrt(dx**2 + dy**2) - r[idx[0]] - r[idx[1]])
    return np.concatenate(c)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    m = n * (n - 1) // 2
    A = np.zeros((m, n))
    b = np.zeros(m)
    idx = np.triu_indices(n, 1)
    A[np.arange(m), idx[0]] = 1.0
    A[np.arange(m), idx[1]] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b[:] = dists[idx[0], idx[1]]
    
    ub = np.array([min(c[0], 1.0 - c[0], c[1], 1.0 - c[1]) for c in centers])
    ub = np.maximum(ub, 1e-9)
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0.0, u) for u in ub], method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def slesqp_unequal(init_c):
    """Refines centers and radii jointly using SLSQP."""
    n = init_c.shape[0]
    x0 = np.zeros(3 * n)
    x0[:n] = init_c[:, 0]
    x0[n:2*n] = init_c[:, 1]
    
    # Start with feasible radii slightly shrunk to ensure constraint satisfaction
    r_est, _ = solve_lp(init_c)
    x0[2*n:] = r_est * 0.95
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    try:
        res = minimize(slesqp_objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': slesqp_constraints, 'args': (n,)},
                      options={'maxiter': 4000, 'ftol': 1e-13})
        if np.isfinite(res.fun):
            return np.column_stack((res.x[:n], res.x[n:2*n])), res.x[2*n:]
    except Exception:
        pass
    return init_c, np.full(n, 0.09)

def force_sim_equal(init_centers, steps=5000):
    """Physics-based simulation to jam equal circles into a dense topology."""
    n = init_centers.shape[0]
    c = init_centers.copy()
    r = 0.03
    vel = np.zeros_like(c)
    dt = 0.005
    damp = 0.85
    
    for _ in range(steps):
        r *= 1.0002  # Gradual expansion
        forces = np.zeros_like(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e9)
        
        overlap = np.maximum(0.0, 2.0 * r - dists)
        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        rep = (50.0 * overlap) / safe_dists
        
        fx = np.sum(diff[:, :, 0] * rep, axis=1)
        fy = np.sum(diff[:, :, 1] * rep, axis=1)
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        for i in range(n):
            if c[i, 0] < r:
                forces[i, 0] += 20.0 * (r - c[i, 0])
            if c[i, 0] > 1.0 - r:
                forces[i, 0] -= 20.0 * (c[i, 0] - (1.0 - r))
            if c[i, 1] < r:
                forces[i, 1] += 20.0 * (r - c[i, 1])
            if c[i, 1] > 1.0 - r:
                forces[i, 1] -= 20.0 * (c[i, 1] - (1.0 - r))
                
        vel = damp * vel + forces * dt
        c += vel
        c = np.clip(c, 0.0, 1.0)
    return c, r

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    # Diverse hexagonal row patterns
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,6,5,5], 
        [6,6,4,6,4], [5,6,6,5,4], [4,5,6,5,6], [6,4,6,5,5],
        [5,5,5,5,6], [6,5,5,5,5], [7,5,5,5,4], [5,7,5,5,4]
    ]
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: Multi-start topology search
    for pat in patterns:
        if sum(pat) != n:
            continue
        pts = []
        y = 0.1
        for i, cnt in enumerate(pat):
            shift = 0.1 if i % 2 == 1 else 0.0
            x = 0.1 + shift
            for _ in range(cnt):
                if len(pts) >= n:
                    break
                pts.append([x, y])
                x += 0.2
            y += 0.1732
        pts = np.array(pts[:n])
        
        # Jam topology with physics simulation
        sim_c, _ = force_sim_equal(pts, steps=4000)
        
        # Refine jointly with SLSQP
        opt_c, opt_r = slesqp_unequal(sim_c)
        
        # Extract exact maximum radii via LP
        lp_r, lp_s = solve_lp(opt_c)
        if lp_s > best_sum:
            best_sum = lp_s
            best_c = opt_c.copy()
            best_r = lp_r.copy()
            
    # Phase 2: Adaptive hill-climbing on centers evaluated via LP
    if best_c is not None:
        for step in range(1500):
            i = rng.integers(n)
            old = best_c[i].copy()
            step_sz = 0.014 * (0.93 ** (step / 35.0))
            best_c[i] += rng.uniform(-step_sz, step_sz, 2)
            best_c[i] = np.clip(best_c[i], 1e-4, 1.0 - 1e-4)
            new_r, new_s = solve_lp(best_c)
            if new_s > best_sum:
                best_sum = new_s
                best_r = new_r.copy()
            else:
                best_c[i] = old
                
    # Phase 3: Strict numerical safeguarding
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_r *= scale * 0.999999
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
