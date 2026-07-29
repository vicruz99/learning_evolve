# sol_000307 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000287 (state 4c08251c) state=d169c0de sum of radii=2.615603 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure globally for speed
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_IDX.append((i, j))
        idx += 1
for i in range(N):
    for _ in range(4):
        A_LP[idx, i] = 1.0
        idx += 1

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes exact gradient via duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    b = np.zeros(A_LP.shape[0])
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(n):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                      bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(n), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(n):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
    return radii, s_sum, grad

def gradient_ascent(centers0, max_iter=1500, init_lr=0.008):
    """Performs adaptive gradient ascent on centers to maximize sum of radii."""
    c = centers0.copy()
    best_c = c.copy()
    best_sum = -1.0
    _, curr_sum, grad = solve_lp_and_grad(c)
    best_sum = curr_sum
    lr = init_lr
    
    for _ in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
        c_new = c + lr * grad / g_norm
        c_new = np.clip(c_new, 0.005, 0.995)
        _, s_new, g_new = solve_lp_and_grad(c_new)
        if s_new > curr_sum:
            c = c_new
            grad = g_new
            curr_sum = s_new
            lr = min(lr * 1.1, 0.04)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = c.copy()
        else:
            lr *= 0.7
        if lr < 1e-8:
            break
    return best_c, best_sum

def simulated_annealing(centers0, max_iter=3000, T0=0.012, rng=None):
    """Simulated annealing on centers with occasional index swapping to break symmetry."""
    if rng is None:
        rng = np.random.default_rng()
    c_curr = centers0.copy()
    _, s_curr, _ = solve_lp_and_grad(c_curr)
    best_c = c_curr.copy()
    best_sum = s_curr
    T = T0
    
    for step in range(max_iter):
        step_size = 0.005 * (T / T0)
        c_try = c_curr + rng.normal(0, step_size, c_curr.shape)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        # Swap two circles to escape symmetric local minima
        if rng.random() < 0.05:
            i, j = rng.choice(N, 2, replace=False)
            c_try[[i, j]] = c_try[[j, i]]
            
        _, s_try, _ = solve_lp_and_grad(c_try)
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c_curr = c_try
            s_curr = s_try
            if s_curr > best_sum:
                best_sum = s_curr
                best_c = c_curr.copy()
        T *= 0.9985
    return best_c, best_sum

def generate_inits(rng):
    """Generates diverse initial configurations: hex grids, corners, force-directed."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [5, 6, 6, 5, 4], [6, 4, 6, 5, 5]
    ]
    for pat in patterns:
        for r_est in [0.098, 0.102, 0.106, 0.110]:
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
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            inits.append(c)
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c[:4] = corners
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        push = (0.22 - d) * 0.04
                        forces[i] += d_vec / d * push
                        forces[j] -= d_vec / d * push
            c += forces
            c = np.clip(c, 0.1, 0.9)
        inits.append(c)
    return inits

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def obj_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_joint_sq(v):
    """Constraints for SLSQP: boundaries and non-overlap using squared distances."""
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

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Multi-start Gradient Ascent
    inits = generate_inits(rng)
    for c0 in inits:
        c_opt, s_opt = gradient_ascent(c0, max_iter=1200, init_lr=0.006)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: Simulated Annealing for Global Exploration
    if best_c is not None:
        c_sa, s_sa = simulated_annealing(best_c, max_iter=4000, T0=0.015, rng=rng)
        if s_sa > best_sum:
            best_sum = s_sa
            best_c = c_sa.copy()
            
        # Phase 3: Refine SA result with Gradient Ascent
        c_opt2, s_opt2 = gradient_ascent(best_c, max_iter=1500, init_lr=0.005)
        if s_opt2 > best_sum:
            best_sum = s_opt2
            best_c = c_opt2.copy()
            
        # Phase 4: Iterative Perturbation & Ascent to escape remaining plateaus
        for step in range(20):
            scale = 0.008 * (0.85 ** (step // 3))
            c_pert = best_c + rng.normal(0, scale, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            c_opt3, s_opt3 = gradient_ascent(c_pert, max_iter=800, init_lr=0.004)
            if s_opt3 > best_sum:
                best_sum = s_opt3
                best_c = c_opt3.copy()

    # Phase 5: SLSQP Joint Polish for Final Precision
    if best_c is not None:
        r_lp, _, _ = solve_lp_and_grad(best_c)
        v0 = np.concatenate([best_c.flatten(), r_lp])
        bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        for _ in range(3):
            try:
                res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_sl,
                              constraints={'type': 'ineq', 'fun': cons_joint_sq},
                              options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                if np.min(cons_joint_sq(res.x)) >= -1e-8:
                    s = np.sum(res.x[2 * N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2 * N].reshape(N, 2).copy()
                        best_r = res.x[2 * N:].copy()
                        v0 = res.x.copy()
            except Exception:
                pass
                
    # Final LP Verification & Strict Repair
    if best_c is not None:
        r_final, s_final, _ = solve_lp_and_grad(best_c)
        if s_final > best_sum:
            best_r = r_final
            best_sum = s_final
        else:
            best_r = r_final
            
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
