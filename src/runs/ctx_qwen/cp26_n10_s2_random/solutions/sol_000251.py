# sol_000251 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000205 (state 0b4dbf91) state=fde5359e sum of radii=2.610168 correctness=1.0
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
    npairs = N * (N - 1) // 2
    A_LP = np.zeros((npairs + 4 * N, N))
    PAIR_IDX = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_LP[k, i] = 1.0
            A_LP[k, j] = 1.0
            PAIR_IDX.append((i, j))
            k += 1
    for i in range(N):
        A_LP[k, i] = 1.0; k += 1
        A_LP[k, i] = 1.0; k += 1
        A_LP[k, i] = 1.0; k += 1
        A_LP[k, i] = 1.0; k += 1

setup_lp()

def lp_radii_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
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
    try:
        duals = np.asarray(res.marginals.ineqlin)
    except AttributeError:
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

def gradient_ascent(c0, max_iter, init_step, rng):
    """Runs gradient ascent on centers to maximize sum of radii."""
    centers = c0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    no_improve = 0
    
    radii, curr_sum, grad = lp_radii_and_grad(centers)
    best_sum = curr_sum
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            centers = centers + step * grad / g_norm
        else:
            step *= 0.5
            
        centers = np.clip(centers, 0.005, 0.995)
        
        radii, curr_sum, grad = lp_radii_and_grad(centers)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            step = min(step * 1.08, 0.025)
            no_improve = 0
        else:
            step *= 0.85
            no_improve += 1
            
        # Periodic jitter to escape local minima
        if k > 0 and k % 250 == 0:
            noise_scale = 0.004 * (0.7 ** (k // 250))
            centers = centers + rng.normal(0, noise_scale, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            radii, curr_sum, grad = lp_radii_and_grad(centers)
            
        if step < 1e-8 or no_improve > 1500:
            break
            
    return best_centers, best_sum

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 6, 4, 5, 6], [5, 5, 4, 6, 6], [6, 6, 4, 5, 5],
        [5, 7, 5, 5, 4], [4, 6, 5, 6, 5], [5, 6, 6, 4, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.092, 0.098, 0.104, 0.110]:
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
            c = c + rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(15):
        starts.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.22 and dist > 1e-5:
                        f = (0.22 - dist) / (dist**2 + 1e-6)
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c = c + forces * 0.005
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
    return starts

def joint_slsqp(c0, r0, rng):
    """Polishes packing with joint SLSQP optimization."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(v):
        return -np.sum(v[2 * N:])
        
    def cons(v):
        c = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
        i, j = np.triu_indices(N, 1)
        dx = c[i, 0] - c[j, 0]
        dy = c[i, 1] - c[j, 1]
        dr = r[i] + r[j]
        con.append(dx**2 + dy**2 - dr**2)
        return np.concatenate(con)
        
    best_v = v0.copy()
    best_s = -np.sum(r0)
    
    for _ in range(8):
        v_trial = v0 + rng.normal(0, 0.0015, v0.shape)
        v_trial = np.clip(v_trial, 0.01, 0.99)
        v_trial[2 * N:] = np.clip(v_trial[2 * N:], 0.01, 0.4)
        try:
            res = minimize(obj, v_trial, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': cons},
                          options={'maxiter': 8000, 'ftol': 1e-13})
            s = np.sum(res.x[2 * N:])
            if s > best_s and np.min(cons(res.x)) >= -1e-8:
                best_s = s
                best_v = res.x.copy()
        except Exception:
            pass
            
    return best_v[:2 * N].reshape(N, 2), best_v[2 * N:], best_s

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
    best_r = None
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient Ascent from multiple starts
    for c0 in starts:
        c_opt, s_opt = gradient_ascent(c0, max_iter=4500, init_step=0.012, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = lp_radii_and_grad(best_c)
            
    # Phase 2: Iterative Shrink-Perturb-Grow to escape local optima
    if best_c is not None:
        for _ in range(6):
            c_pert = best_c + rng.normal(0, 0.005, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            c_opt, s_opt = gradient_ascent(c_pert, max_iter=3000, init_step=0.009, rng=rng)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r, _, _ = lp_radii_and_grad(best_c)
                
        # Phase 3: Joint SLSQP Polish
        c_pol, r_pol, s_pol = joint_slsqp(best_c, best_r, rng)
        if s_pol > best_sum:
            best_sum = s_pol
            best_c = c_pol
            best_r = r_pol
            
    # Phase 4: Final Repair
    centers = best_c.copy()
    radii = repair_packing(centers, best_r.copy())
    
    return centers, radii, float(np.sum(radii))
