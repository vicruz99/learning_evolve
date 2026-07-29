# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000030 (state 57c93ce5) state=6a99b8b1 sum of radii=2.631094 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    i_idx, j_idx = np.tril_indices(n, -1)
    dist_sq = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2
    r_sum_sq = dr[i_idx, j_idx]**2
    
    c = np.concatenate([c, dist_sq - r_sum_sq])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def init_hex(seed, r0):
    """Generates a hexagonal lattice initialization."""
    np.random.seed(seed)
    x = np.zeros(3 * N)
    idx = 0
    y = r0
    row = 0
    while idx < N and y + r0 <= 1.0:
        start_x = r0 if row % 2 == 0 else 2 * r0
        cx = start_x
        while idx < N and cx + r0 <= 1.0:
            x[3 * idx] = cx
            x[3 * idx + 1] = y
            x[3 * idx + 2] = r0
            idx += 1
            cx += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
    # Fill remaining if any
    while idx < N:
        x[3 * idx] = np.random.uniform(0.1, 0.9)
        x[3 * idx + 1] = np.random.uniform(0.1, 0.9)
        x[3 * idx + 2] = 0.05
        idx += 1
    return x

def force_simulate(x0, steps=200):
    """Force-directed relaxation to resolve overlaps and boundary violations."""
    n = N
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for _ in range(steps):
        fx, fy = np.zeros(n), np.zeros(n)
        for i, j in pairs:
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dist = np.hypot(dx, dy)
            if dist < rs[i] + rs[j] and dist > 1e-6:
                force = (rs[i] + rs[j] - dist) * 0.5 / dist
                fx[i] -= force * dx
                fy[i] -= force * dy
                fx[j] += force * dx
                fy[j] += force * dy
        for i in range(n):
            if xs[i] < rs[i]: fx[i] += 0.1 * (rs[i] - xs[i])
            if xs[i] > 1.0 - rs[i]: fx[i] -= 0.1 * (xs[i] - (1.0 - rs[i]))
            if ys[i] < rs[i]: fy[i] += 0.1 * (rs[i] - ys[i])
            if ys[i] > 1.0 - rs[i]: fy[i] -= 0.1 * (ys[i] - (1.0 - ys[i]))
            
        xs += fx * 0.1
        ys += fy * 0.1
        xs = np.clip(xs, 0.0, 1.0)
        ys = np.clip(ys, 0.0, 1.0)
        
    x = np.zeros(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x

def solve_lp_radii(centers):
    """Optimally scale radii for fixed centers using Linear Programming."""
    n = N
    c = -np.ones(n)  # Maximize sum of radii
    
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x_val, y_val = centers[i]
        bound = min(x_val, 1.0 - x_val, y_val, 1.0 - y_val)
        A_ub[idx, i] = 1.0
        b_ub[idx] = bound
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def run_packing():
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multiple diverse starts with force relaxation + SLSQP
    inits = []
    for s in range(15):
        inits.append(init_hex(s, 0.095))
        inits.append(init_hex(s, 0.10))
        
    for x0 in inits:
        x0_pert = x0 + np.random.normal(0, 0.002, 3 * N)
        x0_pert[2::3] = np.clip(x0_pert[2::3], 0.01, 0.4)
        x0_sim = force_simulate(x0_pert, steps=200)
        
        try:
            res = minimize(objective, x0_sim, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            vals = constraints(res.x)
            if np.min(vals) >= -1e-7 and curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Alternating Optimization (SLSQP positions <-> LP radii)
    if best_x is not None:
        for _ in range(4):
            centers = np.zeros((N, 2))
            centers[:, 0] = best_x[0::3]
            centers[:, 1] = best_x[1::3]
            
            new_radii = solve_lp_radii(centers)
            if new_radii is not None:
                best_x[2::3] = new_radii
                curr_sum = np.sum(new_radii)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    
                # Re-optimize positions with updated radii
                x0_fix = best_x.copy()
                x0_fix[0::3] += np.random.normal(0, 1e-5, N)
                x0_fix[1::3] += np.random.normal(0, 1e-5, N)
                
                try:
                    res = minimize(objective, x0_fix, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                    if -res.fun > best_sum:
                        best_x = res.x.copy()
                        best_sum = -res.fun
                except Exception:
                    pass
                    
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    centers[:, 0] = best_x[0::3]
    centers[:, 1] = best_x[1::3]
    radii[:] = best_x[2::3]
    
    # Final safety shrink to guarantee strict compliance with validator
    for _ in range(30):
        valid = True
        for i in range(N):
            if radii[i] < 0: valid = False; break
            if centers[i,0] - radii[i] < -1e-10 or centers[i,0] + radii[i] > 1 + 1e-10: valid = False; break
            if centers[i,1] - radii[i] < -1e-10 or centers[i,1] + radii[i] > 1 + 1e-10: valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                    if d < radii[i] + radii[j] - 1e-10:
                        valid = False; break
                if not valid: break
        if valid: break
        radii *= 0.995
        
    return centers, radii, float(np.sum(radii))
