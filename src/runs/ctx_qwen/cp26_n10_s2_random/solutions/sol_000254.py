# sol_000254 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000205 (state 0b4dbf91) state=14dc934c sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
A_LP = None
PAIR_IDX = None

def setup_lp_matrices():
    """Precompute the constant structure of the LP constraint matrix."""
    global A_LP, PAIR_IDX
    num_pairs = N * (N - 1) // 2
    num_bound = 4 * N
    A_LP = np.zeros((num_pairs + num_bound, N))
    PAIR_IDX = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_LP[k, i] = 1.0
            A_LP[k, j] = 1.0
            PAIR_IDX.append((i, j))
            k += 1
    for i in range(N):
        for _ in range(4):
            A_LP[k, i] = 1.0
            k += 1

setup_lp_matrices()

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b_ub = np.zeros(A_LP.shape[0])
    k = 0
    for i, j in PAIR_IDX:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(N):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    duals = np.zeros(A_LP.shape[0])
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(N):
        mu_x0 = duals[bound_start + 4 * i]
        mu_x1 = duals[bound_start + 4 * i + 1]
        mu_y0 = duals[bound_start + 4 * i + 2]
        mu_y1 = duals[bound_start + 4 * i + 3]
        grad[i, 0] += mu_x0 - mu_x1
        grad[i, 1] += mu_y0 - mu_y1
        
    return radii, np.sum(radii), grad

def obj_grad(x_flat):
    """Objective and gradient for scipy optimizer: minimizes negative sum of radii."""
    c = x_flat.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared distances)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 6, 4, 5, 6], [5, 5, 4, 6, 6], [6, 6, 4, 5, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.08, 0.088, 0.095, 0.102, 0.11]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(12):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.25 and dist > 1e-4:
                        f = (0.25 - dist) / (dist**2 + 1e-6)
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces * 0.005
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Corner-biased starts
    for _ in range(6):
        c = rng.uniform(0.2, 0.8, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        starts.append(c)
        
    return starts

def coordinate_refinement(centers, rng):
    """Refines each circle's position independently using Nelder-Mead."""
    best_c = centers.copy()
    _, best_s, _ = solve_lp_and_grad(best_c)
    
    for _ in range(3):
        for i in range(N):
            def obj_2d(xy):
                temp = best_c.copy()
                temp[i] = np.clip(xy, 0.001, 0.999)
                _, s, _ = solve_lp_and_grad(temp)
                return -s
                
            try:
                res = minimize(obj_2d, best_c[i], method='Nelder-Mead', 
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12})
                if -res.fun > best_s + 1e-9:
                    best_c[i] = np.clip(res.x, 0.001, 0.999)
                    _, best_s, _ = solve_lp_and_grad(best_c)
            except Exception:
                pass
    return best_c

def simulated_annealing(centers, rng):
    """Simulated annealing to escape local optima."""
    c_curr = centers.copy()
    _, s_curr, _ = solve_lp_and_grad(c_curr)
    best_c = c_curr.copy()
    best_s = s_curr
    
    T = 0.006
    decay = 0.995
    
    for step in range(2500):
        i = rng.integers(N)
        c_try = c_curr.copy()
        c_try[i] += rng.normal(0, 0.008, 2)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c_curr = c_try
            s_curr = s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
        T *= decay
        
    return best_c

def repair_packing(centers, radii):
    """Deterministic repair to ensure strict validity."""
    radii = radii.copy()
    for _ in range(60):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-11:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_sum = -1.0
    bounds_centers = [(0.005, 0.995)] * (2 * N)
    
    starts = generate_starts(rng)
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    for c0 in starts:
        try:
            res = minimize(obj_grad, c0.flatten(), method='L-BFGS-B', 
                           bounds=bounds_centers, jac=True,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10})
            _, s, _ = solve_lp_and_grad(res.x.reshape(N, 2))
            if s > best_sum:
                best_sum = s
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        _, best_sum, _ = solve_lp_and_grad(best_c)

    # Phase 2: Coordinate-wise refinement
    best_c = coordinate_refinement(best_c, rng)
    _, best_sum, _ = solve_lp_and_grad(best_c)

    # Phase 3: Simulated Annealing
    best_c = simulated_annealing(best_c, rng)
    _, best_sum, _ = solve_lp_and_grad(best_c)
    
    # Re-optimize best SA result with L-BFGS-B to settle
    try:
        res2 = minimize(obj_grad, best_c.flatten(), method='L-BFGS-B', 
                        bounds=bounds_centers, jac=True,
                        options={'maxiter': 3000, 'ftol': 1e-14})
        _, s2, _ = solve_lp_and_grad(res2.x.reshape(N, 2))
        if s2 > best_sum:
            best_sum = s2
            best_c = res2.x.reshape(N, 2).copy()
    except Exception:
        pass

    # Phase 4: Joint SLSQP Polish
    radii_init, _, _ = solve_lp_and_grad(best_c)
    v0 = np.concatenate([best_c.flatten(), radii_init])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res_sl = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_sl,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 10000, 'ftol': 1e-14})
        if np.min(slsqp_cons(res_sl.x)) >= -1e-7:
            s_sl = np.sum(res_sl.x[2 * N:])
            if s_sl > best_sum:
                best_sum = s_sl
                best_c = res_sl.x[:2 * N].reshape(N, 2)
                radii_init = res_sl.x[2 * N:]
    except Exception:
        pass

    # Phase 5: Deterministic Repair
    centers = best_c.copy()
    radii = repair_packing(centers, radii_init.copy())
    
    return centers, radii, float(np.sum(radii))
