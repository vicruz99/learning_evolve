# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=7c76ac7a sum of radii=2.628522 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective: minimize negative sum of radii (equivalent to maximizing sum of radii)."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Vectorized inequality constraints: >= 0 for valid packing."""
    n = N_CIRCLES
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    rs = r[:, None] + r[None, :]
    iu, ju = np.triu_indices(n, k=1)
    c.append(d[iu, ju] - rs[iu, ju])
    
    return np.concatenate(c)

def check_valid(centers, radii):
    """Strict validation matching the grader's tolerance."""
    n = N_CIRCLES
    x, y = centers[:, 0], centers[:, 1]
    r = radii
    
    if np.any(r < -1e-12):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1.0 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1.0 + 1e-9):
        return False
        
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    r_sums = r[:, None] + r[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.all(dists[mask] >= r_sums[mask] - 1e-9)

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    inits = []
    
    # --- 1. Force-Simulated Initialization ---
    # Start with hex grid and let repulsive forces spread & expand circles
    r0 = 0.10
    pts = []
    y = r0
    row = 0
    while y < 1.0 and len(pts) < n + 5:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x < 1.0 and len(pts) < n + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    pts = np.array(pts[:n])
    
    centers_sim = pts.copy()
    radii_sim = np.full(n, 0.09)
    dt = 0.002
    
    for step in range(800):
        forces = np.zeros_like(centers_sim)
        # Boundary repulsion
        for i in range(n):
            cx, cy = centers_sim[i]
            cr = radii_sim[i]
            if cx < cr: forces[i, 0] += 20.0 * (cr - cx)
            if cx > 1-cr: forces[i, 0] -= 20.0 * (cx + cr - 1)
            if cy < cr: forces[i, 1] += 20.0 * (cr - cy)
            if cy > 1-cr: forces[i, 1] -= 20.0 * (cy + cr - 1)
            
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_sim[j, 0] - centers_sim[i, 0]
                dy = centers_sim[j, 1] - centers_sim[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6: dist = 1e-6
                req = radii_sim[i] + radii_sim[j]
                if dist < req:
                    overlap = req - dist
                    fx = (dx/dist) * overlap
                    fy = (dy/dist) * overlap
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
                    
        centers_sim += forces * dt
        centers_sim = np.clip(centers_sim, 1e-4, 1.0-1e-4)
        radii_sim += 0.00005
        radii_sim = np.clip(radii_sim, 0.05, 0.5)
        
    v0 = np.zeros(3 * n)
    v0[0::3] = centers_sim[:, 0]
    v0[1::3] = centers_sim[:, 1]
    v0[2::3] = radii_sim
    inits.append(v0)
    
    # --- 2. Random Perturbations ---
    for _ in range(8):
        v = v0.copy()
        v[0::3] += np.random.uniform(-0.04, 0.04, n)
        v[1::3] += np.random.uniform(-0.04, 0.04, n)
        v[0::3] = np.clip(v[0::3], 0.05, 0.95)
        v[1::3] = np.clip(v[1::3], 0.05, 0.95)
        inits.append(v)
        
    # --- 3. Regular Grid Fallback ---
    v = np.zeros(3 * n)
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < n:
                v[3*idx] = 0.1 + j*0.18
                v[3*idx+1] = 0.1 + i*0.18
                v[3*idx+2] = 0.08
                idx += 1
    inits.append(v)
    
    # --- Optimization Loop ---
    for v_init in inits:
        res = minimize(objective, v_init, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
        
        if res.success:
            x_o = res.x[0::3]
            y_o = res.x[1::3]
            r_o = res.x[2::3]
            
            valid = check_valid(np.column_stack((x_o, y_o)), r_o)
            if not valid:
                while not valid and np.mean(r_o) > 0.01:
                    r_o *= 0.99
                    valid = check_valid(np.column_stack((x_o, y_o)), r_o)
            
            if valid:
                s = np.sum(r_o)
                if s > best_sum:
                    best_sum = s
                    best_centers = np.column_stack((x_o, y_o))
                    best_radii = r_o.copy()
                    
            # Local perturbation search around optimum
            for _ in range(5):
                v_p = res.x.copy()
                v_p[0::3] += np.random.uniform(-0.005, 0.005, n)
                v_p[1::3] += np.random.uniform(-0.005, 0.005, n)
                v_p[0::3] = np.clip(v_p[0::3], 0.0, 1.0)
                v_p[1::3] = np.clip(v_p[1::3], 0.0, 1.0)
                
                res2 = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                                constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12})
                if res2.success:
                    x2 = res2.x[0::3]
                    y2 = res2.x[1::3]
                    r2 = res2.x[2::3]
                    valid2 = check_valid(np.column_stack((x2, y2)), r2)
                    if not valid2:
                        while not valid2 and np.mean(r2) > 0.01:
                            r2 *= 0.99
                            valid2 = check_valid(np.column_stack((x2, y2)), r2)
                    if valid2:
                        s2 = np.sum(r2)
                        if s2 > best_sum:
                            best_sum = s2
                            best_centers = np.column_stack((x2, y2))
                            best_radii = r2.copy()

    # Fallback if all optimizations fail
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, 0.08)
        k = 0
        for i in range(6):
            for j in range(5):
                if k >= n: break
                best_centers[k] = [0.1 + j*0.18, 0.1 + i*0.18]
                k += 1
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
