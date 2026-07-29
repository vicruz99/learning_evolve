# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000056 (state 6a99b8b1) state=d8a6d073 sum of radii=2.502578 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def con_fun(x):
    """Inequality constraints g(x) >= 0."""
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    nc = 4 * n + n * (n - 1) // 2
    c = np.empty(nc)
    idx = 0
    for i in range(n):
        c[idx] = xs[i] - rs[i]; idx += 1
        c[idx] = 1.0 - xs[i] - rs[i]; idx += 1
        c[idx] = ys[i] - rs[i]; idx += 1
        c[idx] = 1.0 - ys[i] - rs[i]; idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            rs_sum = rs[i] + rs[j]
            c[idx] = dx*dx + dy*dy - rs_sum*rs_sum
            idx += 1
    return c

def con_jac(x):
    """Jacobian of inequality constraints."""
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    nc = 4 * n + n * (n - 1) // 2
    J = np.zeros((nc, 3 * n))
    idx = 0
    for i in range(n):
        J[idx, 3*i] = 1.0; J[idx, 3*i+2] = -1.0; idx += 1
        J[idx, 3*i] = -1.0; J[idx, 3*i+2] = -1.0; idx += 1
        J[idx, 3*i+1] = 1.0; J[idx, 3*i+2] = -1.0; idx += 1
        J[idx, 3*i+1] = -1.0; J[idx, 3*i+2] = -1.0; idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            rs_sum = rs[i] + rs[j]
            d_dx = 2.0 * dx
            d_dy = 2.0 * dy
            d_dr = -2.0 * rs_sum
            J[idx, 3*i] = d_dx; J[idx, 3*i+1] = d_dy; J[idx, 3*i+2] = d_dr
            J[idx, 3*j] = -d_dx; J[idx, 3*j+1] = -d_dy; J[idx, 3*j+2] = d_dr
            idx += 1
    return J

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
    while idx < N:
        x[3 * idx] = np.random.uniform(0.1, 0.9)
        x[3 * idx + 1] = np.random.uniform(0.1, 0.9)
        x[3 * idx + 2] = 0.05
        idx += 1
    return x

def init_grid(seed):
    """Generates a perturbed grid initialization."""
    np.random.seed(seed)
    x = np.zeros(3 * N)
    idx = 0
    for r in range(6):
        for c in range(5):
            if idx < N:
                x[3*idx] = 0.1 + c * 0.18
                x[3*idx+1] = 0.1 + r * 0.16
                x[3*idx+2] = 0.09
                idx += 1
    while idx < N:
        x[3 * idx] = np.random.uniform(0.1, 0.9)
        x[3 * idx + 1] = np.random.uniform(0.1, 0.9)
        x[3 * idx + 2] = 0.05
        idx += 1
    x += np.random.uniform(-0.005, 0.005, 3 * N)
    x[2::3] = np.clip(x[2::3], 0.01, 0.4)
    return x

def force_simulate(x0, steps=150):
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
                force = (rs[i] + rs[j] - dist) * 0.8 / dist
                fx[i] -= force * dx
                fy[i] -= force * dy
                fx[j] += force * dx
                fy[j] += force * dy
        for i in range(n):
            m = rs[i] + 0.005
            if xs[i] < m: fx[i] += 0.3 * (m - xs[i])
            if xs[i] > 1.0 - m: fx[i] -= 0.3 * (xs[i] - (1.0 - m))
            if ys[i] < m: fy[i] += 0.3 * (m - ys[i])
            if ys[i] > 1.0 - m: fy[i] -= 0.3 * (ys[i] - (1.0 - m))
            
        xs += fx * 0.05
        ys += fy * 0.05
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
    c = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        bound = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(bound, 0.0)
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(dist, 0.0)
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
    cons = {'type': 'ineq', 'fun': con_fun, 'jac': con_jac}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multiple diverse starts with force relaxation + SLSQP
    inits = []
    for s in range(15):
        inits.append(init_hex(s, 0.095))
        inits.append(init_hex(s, 0.105))
    for s in range(10):
        inits.append(init_grid(s))
        
    np.random.seed(123)
    for _ in range(10):
        x_r = np.random.uniform(0.15, 0.85, 3 * N)
        x_r[2::3] = np.random.uniform(0.06, 0.12, N)
        inits.append(x_r)
        
    for x0 in inits:
        x0_pert = x0 + np.random.normal(0, 0.001, 3 * N)
        x0_pert[2::3] = np.clip(x0_pert[2::3], 0.01, 0.4)
        x0_sim = force_simulate(x0_pert, steps=100)
        
        try:
            res = minimize(objective, x0_sim, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            c_vals = con_fun(res.x)
            if np.min(c_vals) >= -1e-7 and curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Alternating LP radii + SLSQP centers + Shrink-Expand refinement
    if best_x is not None:
        for step in range(8):
            centers = np.column_stack((best_x[0::3], best_x[1::3]))
            new_radii = solve_lp_radii(centers)
            if new_radii is not None:
                best_x[2::3] = new_radii
                curr_sum = np.sum(new_radii)
                
                # Perturb centers slightly to explore neighborhood
                x_pert = best_x + np.random.normal(0, 0.0008, 3 * N)
                x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
                x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
                
                try:
                    res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                    if -res.fun > best_sum:
                        best_x = res.x.copy()
                        best_sum = -res.fun
                except Exception:
                    pass
                    
                # Shrink-expand trick to escape local minima
                if step % 2 == 1:
                    best_x[2::3] *= 0.95
                    x_shr = best_x + np.random.normal(0, 0.001, 3 * N)
                    x_shr[0::3] = np.clip(x_shr[0::3], 0.01, 0.99)
                    x_shr[1::3] = np.clip(x_shr[1::3], 0.01, 0.99)
                    try:
                        res = minimize(objective, x_shr, method='SLSQP', bounds=bounds,
                                       constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
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
    
    # Final safety check & minimal shrink to guarantee strict compliance
    for _ in range(50):
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
        radii *= 0.998
        for i in range(N):
            centers[i,0] = np.clip(centers[i,0], radii[i], 1.0 - radii[i])
            centers[i,1] = np.clip(centers[i,1], radii[i], 1.0 - radii[i])
            
    return centers, radii, float(np.sum(radii))
