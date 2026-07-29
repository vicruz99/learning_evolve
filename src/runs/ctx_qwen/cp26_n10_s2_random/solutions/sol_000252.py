# sol_000252 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000205 (state 0b4dbf91) state=c2c756aa sum of radii=2.614591 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
A_LP = None
PAIR_IDX = None

def setup_lp():
    """Precompute the sparse structure of the LP constraint matrix."""
    global A_LP, PAIR_IDX
    num_pairs = N * (N - 1) // 2
    A_LP = np.zeros((num_pairs + 4 * N, N))
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

setup_lp()

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                      bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    try:
        duals = np.asarray(res.ineqlin.marginals)
    except Exception:
        duals = np.zeros(A_LP.shape[0])
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-7:
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

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 6, 4, 5, 6], [6, 5, 5, 6, 4], [5, 5, 4, 6, 6]
    ]
    r_est_list = [0.085, 0.090, 0.095, 0.100, 0.105]
    
    for pat in patterns:
        for r_est in r_est_list:
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
            c = np.clip(c, 0.02, 0.98)
            starts.append(c)
            
    for _ in range(12):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(300):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.20 and dist > 1e-4:
                        f = (0.20 - dist) / (dist**2 + 1e-6)
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces * 0.004
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def gradient_ascent(c0, rng, max_iter=1500):
    """Runs gradient ascent on centers to maximize sum of radii."""
    c = c0.copy()
    best_c = c.copy()
    best_sum = -1.0
    radii, curr_sum, grad = solve_lp_and_grad(c)
    if curr_sum > 0:
        best_sum = curr_sum
    
    step = 0.012
    for k in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            direction = grad / g_norm
        else:
            direction = np.zeros_like(c)
            
        c_new = c + step * direction
        c_new = np.clip(c_new, 0.005, 0.995)
        
        r_new, s_new, g_new = solve_lp_and_grad(c_new)
        
        if s_new > curr_sum:
            c = c_new
            curr_sum = s_new
            grad = g_new
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = c.copy()
            step = min(step * 1.12, 0.025)
        else:
            step *= 0.65
            if step < 1e-8:
                break
                
        if k % 250 == 0 and k > 0:
            noise_scale = 0.004 * (0.75 ** (k // 250))
            c = np.clip(c + rng.normal(0, noise_scale, c.shape), 0.01, 0.99)
            r_new, s_new, g_new = solve_lp_and_grad(c)
            curr_sum = s_new
            grad = g_new
            
    return best_c, best_sum

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared)."""
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

def repair_packing(centers, radii):
    """Deterministic repair to ensure strict validity."""
    radii = radii.copy()
    for _ in range(80):
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
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient ascent from diverse starts
    for c0 in starts:
        c_opt, s_opt = gradient_ascent(c0, rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        # Phase 2: Simulated Annealing with cluster moves
        c_curr = best_c.copy()
        s_curr = best_sum
        T = 0.006
        decay = 0.994
        
        for step in range(2500):
            n_move = rng.integers(2, 10)
            idx = rng.choice(N, n_move, replace=False)
            c_try = c_curr.copy()
            c_try[idx] += rng.normal(0, T, (n_move, 2))
            c_try = np.clip(c_try, 0.01, 0.99)
            
            _, s_try, _ = solve_lp_and_grad(c_try)
            
            delta = s_try - s_curr
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
                c_curr = c_try
                s_curr = s_try
                if s_curr > best_sum:
                    best_sum = s_curr
                    best_c = c_curr.copy()
            T *= decay
            
        # Re-optimize best SA result with gradient ascent
        c_opt, s_opt = gradient_ascent(best_c, rng, max_iter=1500)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()

    # Phase 3: Joint SLSQP Polish
    radii_init, _, _ = solve_lp_and_grad(best_c)
    v0 = np.concatenate([best_c.flatten(), radii_init])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res_sl = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_sl,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 12000, 'ftol': 1e-14})
        if np.min(slsqp_cons(res_sl.x)) >= -1e-7:
            s_sl = np.sum(res_sl.x[2 * N:])
            if s_sl > best_sum:
                best_sum = s_sl
                best_c = res_sl.x[:2 * N].reshape(N, 2)
                radii_init = res_sl.x[2 * N:]
    except Exception:
        pass

    # Phase 4: Deterministic Repair
    centers = best_c.copy()
    radii = repair_packing(centers, radii_init.copy())
    
    return centers, radii, float(np.sum(radii))
