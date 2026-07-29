# sol_000312 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000286 (state d00da21c) state=931936f4 sum of radii=2.619479 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, 1)
NUM_PAIRS = len(PAIR_I)
NUM_BOUNDS = 4 * N
NUM_CON = NUM_PAIRS + NUM_BOUNDS

# Precompute constant A_ub matrix for LP
A_LP = np.zeros((NUM_CON, N))
A_LP[:NUM_PAIRS, PAIR_I] = 1.0
A_LP[:NUM_PAIRS, PAIR_J] = 1.0
for i in range(N):
    A_LP[NUM_PAIRS + 4*i, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 1, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 2, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii and computes gradient w.r.t centers."""
    n = N
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
    b_ub = np.zeros(NUM_CON)
    b_ub[:NUM_PAIRS] = dists[PAIR_I, PAIR_J]
    
    xb = centers[:, 0]
    yb = centers[:, 1]
    b_ub[NUM_PAIRS : NUM_PAIRS + N] = xb
    b_ub[NUM_PAIRS + N : NUM_PAIRS + 2*N] = 1.0 - xb
    b_ub[NUM_PAIRS + 2*N : NUM_PAIRS + 3*N] = yb
    b_ub[NUM_PAIRS + 3*N : NUM_PAIRS + 4*N] = 1.0 - yb
    
    ub_r = np.minimum(np.minimum(xb, 1.0 - xb), np.minimum(yb, 1.0 - yb))
    ub_r = np.maximum(ub_r, 1e-9)
    
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0, u) for u in ub_r], method='highs')
        if not res.success:
            return np.full(n, 1e-4), 0.0, np.zeros_like(centers)
            
        radii = res.x
        sum_r = np.sum(radii)
        
        duals = np.zeros(NUM_CON)
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except AttributeError:
            try:
                duals = np.asarray(res.ineqlin.marginals)
            except AttributeError:
                pass
                
        grad = np.zeros_like(centers)
        # Pairwise constraints gradients
        for k in range(NUM_PAIRS):
            lam = duals[k]
            if lam > 1e-8:
                i, j = PAIR_I[k], PAIR_J[k]
                d = dists[i, j]
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
                    
        # Boundary constraints gradients
        for i in range(N):
            base = NUM_PAIRS + 4 * i
            mu_L = duals[base]
            mu_R = duals[base + 1]
            mu_B = duals[base + 2]
            mu_T = duals[base + 3]
            grad[i, 0] += mu_L - mu_R
            grad[i, 1] += mu_B - mu_T
            
        return radii, sum_r, grad
    except Exception:
        return np.full(n, 1e-4), 0.0, np.zeros_like(centers)

def lbfgs_obj_grad(x_flat):
    centers = np.clip(x_flat.reshape(N, 2), 1e-4, 1.0 - 1e-4)
    _, val, grad = solve_lp_and_grad(centers)
    return -val, -grad.flatten()

def gradient_ascent(c0, max_iter=3000, init_step=0.006):
    c = c0.copy()
    best_c = c.copy()
    best_sum = -1.0
    step = init_step
    momentum = np.zeros_like(c)
    
    _, curr_sum, grad = solve_lp_and_grad(c)
    best_sum = curr_sum
    
    for _ in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-12:
            break
            
        momentum = 0.6 * momentum + 0.4 * grad
        g_dir = momentum / (np.linalg.norm(momentum) + 1e-12)
        
        c_new = c + step * g_dir
        c_new = np.clip(c_new, 0.001, 0.999)
        
        _, s_new, g_new = solve_lp_and_grad(c_new)
        
        if s_new > curr_sum:
            c = c_new
            curr_sum = s_new
            grad = g_new
            step = min(step * 1.05, 0.03)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = c.copy()
        else:
            step *= 0.85
            
        if step < 1e-9:
            break
    return best_c, best_sum

def generate_starts(rng):
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6]
    ]
    
    for pat in patterns:
        for r_est in [0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            while len(c) < N:
                c.append(rng.uniform(0.2, 0.8, 2))
            starts.append(np.array(c[:N]))
            
    # Force-directed spreads
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        push = (0.22 - d) * 0.05
                        forces[i] += d_vec / d * push
                        forces[j] -= d_vec / d * push
            c += forces
            c = np.clip(c, 0.1, 0.9)
        starts.append(c)
        
    return starts

def obj_joint(v):
    return -np.sum(v[2 * N:])

def cons_joint_sq(v):
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def repair(centers, radii):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.001, 0.999)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start optimization (L-BFGS-B and Gradient Ascent)
    for c0 in starts:
        # L-BFGS-B
        try:
            res = minimize(lbfgs_obj_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 2000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
        except Exception:
            pass
            
        # Gradient Ascent
        c_ga, s_ga = gradient_ascent(c0, max_iter=2500, init_step=0.008)
        if s_ga > best_sum:
            best_sum = s_ga
            best_c = c_ga.copy()
            
    if best_c is None:
        best_c = starts[0]
    best_r, best_sum, _ = solve_lp_and_grad(best_c)
    
    # Phase 2: Simulated Annealing with Subset Perturbations & Swaps
    T = 0.006
    for step in range(100):
        noise_scale = 0.005 * (1.0 + 0.3 * np.exp(-step / 15.0))
        
        # Decide perturbation type
        p_type = rng.random()
        c_pert = best_c.copy()
        
        if p_type < 0.4:
            # Full vector perturbation
            c_pert += rng.normal(0, noise_scale, c_pert.shape)
        elif p_type < 0.7:
            # Subset perturbation (move 3-6 circles)
            k = rng.integers(3, 7)
            idxs = rng.choice(N, k, replace=False)
            c_pert[idxs] += rng.normal(0, noise_scale * 1.5, (k, 2))
        else:
            # Swap two circles
            i, j = rng.choice(N, 2, replace=False)
            c_pert[i], c_pert[j] = c_pert[j].copy(), c_pert[i].copy()
            c_pert += rng.normal(0, noise_scale * 0.5, c_pert.shape)
            
        c_pert = np.clip(c_pert, 0.01, 0.99)
        
        # Local refinement
        c_opt, s_opt = gradient_ascent(c_pert, max_iter=800, init_step=0.005)
        
        delta = s_opt - best_sum
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            if delta > 0:
                T = min(T * 1.1, 0.02)
            else:
                T *= 0.96
                
    # Phase 3: Joint SLSQP Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(3):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_slsqp,
                          constraints={'type': 'ineq', 'fun': cons_joint_sq},
                          options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if np.min(cons_joint_sq(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Final LP verification
    lp_r, lp_sum, _ = solve_lp_and_grad(best_c)
    if lp_sum > best_sum:
        best_r = lp_r
        best_sum = lp_sum
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
