# sol_000260 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000205 (state 0b4dbf91) state=ba3dfc9a sum of radii=2.602055 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
A_LP = None
PAIR_IDX = None

def setup_lp_matrices():
    """Precompute the sparse structure of the LP constraint matrix."""
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
    """Solves LP for optimal radii given fixed centers and computes exact gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
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
    
    # Extract dual marginals robustly
    duals = np.zeros(b.shape[0])
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = np.asarray(res.marginals.ineqlin)
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = np.asarray(res.ineqlin.marginals)
        
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

def gradient_ascent(centers, rng, max_iter=1500, init_step=0.006):
    """Adaptive gradient ascent to maximize sum of radii."""
    c = centers.copy()
    r, s, g = solve_lp_and_grad(c)
    best_c, best_s = c.copy(), s
    step = init_step
    no_improve = 0
    
    for k in range(max_iter):
        gn = np.linalg.norm(g)
        if gn > 1e-9:
            c_new = c + step * (g / gn)
            c_new = np.clip(c_new, 1e-6, 1.0 - 1e-6)
            r_new, s_new, g_new = solve_lp_and_grad(c_new)
            
            if s_new > s + 1e-10:
                c, r, s, g = c_new, r_new, s_new, g_new
                no_improve = 0
                if s > best_s:
                    best_s = s
                    best_c = c.copy()
                step = min(step * 1.08, 0.025)
            else:
                no_improve += 1
                if no_improve > 50:
                    step *= 0.65
                elif no_improve > 20:
                    step *= 0.85
                    
        if step < 1e-9:
            break
            
        # Periodic jitter to escape plateaus
        if k > 0 and k % 300 == 0:
            noise_scale = 0.003 * np.exp(-k / 4000.0)
            c += rng.normal(0, noise_scale, c.shape)
            c = np.clip(c, 1e-6, 1.0 - 1e-6)
            r, s, g = solve_lp_and_grad(c)
            if s > best_s:
                best_s = s
                best_c = c.copy()
                
    return best_c, best_s

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

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 6, 4, 5, 6], [5, 5, 4, 6, 6], [6, 6, 4, 5, 5],
        [5, 7, 5, 5, 4], [6, 6, 6, 4, 4], [4, 6, 5, 6, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.092, 0.098, 0.105]:
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
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.04, 0.96)
            starts.append(c)
            
    # Random dense starts
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Force-directed repulsion starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.22 and dist > 1e-5:
                        f = (0.22 - dist) / (dist**2 + 1e-5)
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces * 0.008
            c = np.clip(c, 0.04, 0.96)
        starts.append(c)
        
    return starts

def repair_packing(centers, radii):
    """Deterministic repair to ensure strict validity."""
    radii = radii.copy()
    for _ in range(80):
        changed = False
        # Boundary constraints
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr - 1e-12, 0.0)
                changed = True
                
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if req > d - 1e-11:
                    shrink = (req - d) * 0.5 + 1e-12
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        if not changed:
            break
    return np.maximum(radii, 1e-9)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_sum = -1.0
    bounds_centers = [(0.01, 0.99)] * (2 * N)
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient ascent from diverse starts
    for c0 in starts:
        try:
            c_opt, s_opt = gradient_ascent(c0, rng, max_iter=1500, init_step=0.007)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Simulated Annealing on best configuration
    if best_c is not None:
        c_curr = best_c.copy()
        s_curr = best_sum
        T = 0.008
        decay = 0.997
        
        for step in range(4000):
            i = rng.integers(N)
            c_try = c_curr.copy()
            c_try[i] += rng.normal(0, 0.007, 2)
            c_try = np.clip(c_try, 0.01, 0.99)
            
            _, s_try, _ = solve_lp_and_grad(c_try)
            
            delta = s_try - s_curr
            if delta > 0 or rng.random() < np.exp(delta / T):
                c_curr = c_try
                s_curr = s_try
                if s_curr > best_sum:
                    best_sum = s_curr
                    best_c = c_curr.copy()
            T *= decay
            
        # Refine SA result
        c_ref, s_ref = gradient_ascent(best_c, rng, max_iter=1000, init_step=0.005)
        if s_ref > best_sum:
            best_sum = s_ref
            best_c = c_ref.copy()

    # Phase 3: Perturbation & Re-optimization loop
    for _ in range(15):
        c_pert = best_c + rng.normal(0, 0.004, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        c_opt, s_opt = gradient_ascent(c_pert, rng, max_iter=800, init_step=0.004)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()

    # Phase 4: Joint SLSQP Polish
    radii_init, _, _ = solve_lp_and_grad(best_c)
    v0 = np.concatenate([best_c.flatten(), radii_init])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res_sl = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_sl,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
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
