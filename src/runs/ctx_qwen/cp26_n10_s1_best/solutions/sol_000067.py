# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000056 (state 6a99b8b1) state=4f336c07 sum of radii=2.634292 correctness=1.0
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

def solve_lp_radii(centers):
    """Optimally scale radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        x_val, y_val = centers[i]
        bound = min(x_val, 1.0 - x_val, y_val, 1.0 - y_val)
        A_ub[idx, i] = 1.0
        b_ub[idx] = bound
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def force_directed_init(n, seed, steps=600):
    """Force-directed relaxation to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (n, 2))
    radii = np.full(n, 0.05)
    
    for step in range(steps):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j] - centers[i]
                dist = np.linalg.norm(dx)
                if dist < 0.25 and dist > 1e-6:
                    f = 0.015 / (dist**2 + 0.001)
                    forces[i] -= f * dx
                    forces[j] += f * dx
                    
            for dim in range(2):
                if centers[i, dim] < radii[i] + 0.05:
                    forces[i, dim] += 0.08
                elif centers[i, dim] > 1.0 - radii[i] - 0.05:
                    forces[i, dim] -= 0.08
                    
        learning_rate = 0.05 * (1.0 - step / steps)
        centers += forces * learning_rate
        centers = np.clip(centers, 0.02, 0.98)
        
    return centers

def run_packing():
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initializations
    inits = []
    
    # Hexagonal lattices with varying parameters
    for r0 in np.linspace(0.085, 0.105, 5):
        for ang in [0.0, 0.15, 0.30, 0.45]:
            centers = np.zeros((N, 2))
            idx = 0
            y = r0
            row = 0
            while idx < N and y + r0 <= 1.0:
                start_x = r0 if row % 2 == 0 else 2.0 * r0
                cx = start_x
                while idx < N and cx + r0 <= 1.0:
                    centers[idx] = [cx, y]
                    idx += 1
                    cx += 2.0 * r0
                y += np.sqrt(3.0) * r0
                row += 1
            while idx < N:
                centers[idx] = np.random.uniform(0.1, 0.9, 2)
                idx += 1
                
            if ang != 0.0:
                c, s = np.cos(ang), np.sin(ang)
                centers = (centers - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            inits.append(centers)
            
    # Force-directed layouts
    for s in range(8):
        inits.append(force_directed_init(N, seed=s))
        
    # Optimize each initialization
    for c_init in inits:
        # Use LP to find optimal radii for these centers
        r_lp = solve_lp_radii(c_init)
        if r_lp is None:
            r_lp = np.full(N, 0.08)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_lp
        
        # Slight perturbation to break exact symmetries
        x0 += np.random.normal(0, 1e-4, 3 * N)
        
        # Ensure bounds are respected
        for i in range(N):
            r = max(0.0, x0[3*i+2])
            x0[3*i] = np.clip(x0[3*i], r, 1.0 - r)
            x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0 - r)
            x0[3*i+2] = r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            vals = constraints(res.x)
            if np.min(vals) >= -1e-6 and curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative Refinement (Deflation + LP + SLSQP)
    if best_x is not None:
        for round_idx in range(25):
            # Deflate radii to allow centers to move
            x0 = best_x.copy()
            x0[2::3] *= 0.985
            
            # Perturb
            noise_scale = 0.002 * (0.92 ** round_idx)
            x0 += np.random.normal(0, noise_scale, 3 * N)
            
            for i in range(N):
                r = max(0.01, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0 - r)
                x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0 - r)
                x0[3*i+2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                
                if -res.fun > best_sum:
                    best_x = res.x.copy()
                    best_sum = -res.fun
                    
                # LP refinement on current centers
                centers_cur = best_x.reshape(N, 3)[:, :2]
                r_lp = solve_lp_radii(centers_cur)
                if r_lp is not None:
                    best_x[2::3] = r_lp
                    curr_sum = np.sum(r_lp)
                    if curr_sum > best_sum:
                        best_sum = curr_sum
            except Exception:
                pass
                
    # Extract results
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x[2::3]
    
    # Final safety adjustment to guarantee strict compliance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i,0] < radii[i]-1e-9 or centers[i,0] > 1-radii[i]+1e-9 or \
               centers[i,1] < radii[i]-1e-9 or centers[i,1] > 1-radii[i]+1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    if np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) < radii[i]+radii[j]-1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        radii *= 0.9995
        for i in range(N):
            centers[i,0] = np.clip(centers[i,0], radii[i], 1-radii[i])
            centers[i,1] = np.clip(centers[i,1], radii[i], 1-radii[i])
            
    return centers, radii, float(np.sum(radii))
