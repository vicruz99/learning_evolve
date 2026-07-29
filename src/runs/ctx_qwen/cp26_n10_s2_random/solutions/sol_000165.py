# sol_000165 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000160 (state 08773110) state=a71a7ee7 sum of radii=2.310025 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute LP constraint matrix structure globally for speed
A_LP = None
PAIR_IDXS = None

def _init_lp_matrix():
    global A_LP, PAIR_IDXS
    n_pairs = N * (N - 1) // 2
    n_bounds = 4 * N
    A_LP = np.zeros((n_pairs + n_bounds, N))
    PAIR_IDXS = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_LP[idx, i] = 1.0
            A_LP[idx, j] = 1.0
            PAIR_IDXS.append((i, j))
            idx += 1
    for i in range(N):
        for _ in range(4):
            A_LP[idx, i] = 1.0
            idx += 1

_init_lp_matrix()

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii and computes exact gradient using dual marginals."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(A_LP.shape[0])
    b_ub[:len(PAIR_IDXS)] = dists[np.triu_indices(n, k=1)]
    
    for i in range(n):
        b_ub[len(PAIR_IDXS) + 4*i] = centers[i, 0]
        b_ub[len(PAIR_IDXS) + 4*i + 1] = 1.0 - centers[i, 0]
        b_ub[len(PAIR_IDXS) + 4*i + 2] = centers[i, 1]
        b_ub[len(PAIR_IDXS) + 4*i + 3] = 1.0 - centers[i, 1]
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, bounds=[(0, 0.5)]*n, method='highs')
        if not res.success:
            return np.zeros(n), 0.0, np.zeros_like(centers)
            
        radii = res.x
        duals = np.asarray(res.ineqlin.marginals).ravel()
        grad = np.zeros_like(centers)
        
        # Pairwise contact forces
        for k, (i, j) in enumerate(PAIR_IDXS):
            lam = duals[k]
            if lam > 1e-7:
                d = dists[i, j]
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
                    
        # Boundary wall forces
        n_pairs = len(PAIR_IDXS)
        for i in range(n):
            mu_L = duals[n_pairs + 4*i]
            mu_R = duals[n_pairs + 4*i + 1]
            mu_B = duals[n_pairs + 4*i + 2]
            mu_T = duals[n_pairs + 4*i + 3]
            grad[i, 0] += mu_L - mu_R
            grad[i, 1] += mu_B - mu_T
            
        return radii, np.sum(radii), grad
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)

def optimize_gradient_ascent(centers0, max_iter=1500, init_step=0.01):
    """Gradient ascent on centers to maximize LP-sum of radii."""
    rng = np.random.default_rng(123)
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    no_improve = 0
    
    for k in range(max_iter):
        radii, curr_sum, grad = solve_lp_and_grad(centers)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        # Adaptive step decay
        if no_improve > 50:
            step *= 0.6
        elif no_improve > 20:
            step *= 0.85
            
        if step < 1e-9:
            break
            
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            centers += step * (grad / g_norm)
            
        # Periodic jitter to escape plateaus
        if k % 100 == 0 and k > 0:
            centers += rng.normal(0, step * 0.5, centers.shape)
            
        centers = np.clip(centers, 0.03, 0.97)
        
    return best_centers, best_sum

def obj_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Inequality constraints for SLSQP (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
    i, j = np.triu_indices(N, 1)
    dx = c[i,0] - c[j,0]
    dy = c[i,1] - c[j,1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def optimize_slsqp(c_init, r_init):
    """Polishes configuration using SLSQP on joint variables."""
    v0 = np.concatenate([c_init.flatten(), r_init])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 3000, 'ftol': 1e-14})
        if res.success and np.min(cons_joint(res.x)) >= -1e-8:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:]
    except Exception:
        pass
    return c_init, r_init

def generate_starts(rng):
    """Generates diverse hexagonal and random initial configurations."""
    starts = []
    patterns = [[5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [4,6,6,6,4], [6,6,5,5,4], [5,5,5,6,5]]
    for pat in patterns:
        for r0 in [0.09, 0.095, 0.10]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            starts.append(np.array(c[:N]))
    for _ in range(8):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation tolerance."""
    radii = radii.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-10:
                    shrink = (req - d)/2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-10:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient Ascent from diverse starts
    for c_init in starts:
        c_opt, s_opt = optimize_gradient_ascent(c_init)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: SLSQP Polish & LP Refinement
    if best_c is not None:
        r_lp, _, _ = solve_lp_and_grad(best_c)
        c_s, r_s = optimize_slsqp(best_c, r_lp)
        s_s = np.sum(r_s)
        if s_s > best_sum:
            best_sum = s_s
            best_c = c_s
            best_r = r_s
        else:
            best_r = r_lp
            
        # Phase 3: Perturbation & Re-optimization to escape local minima
        for _ in range(5):
            c_per = best_c + rng.normal(0, 0.003, best_c.shape)
            c_per = np.clip(c_per, 0.05, 0.95)
            c_opt, s_opt = optimize_gradient_ascent(c_per, max_iter=800, init_step=0.005)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                r_lp, _, _ = solve_lp_and_grad(best_c)
                best_r = r_lp
                
            c_s, r_s = optimize_slsqp(best_c, best_r)
            if np.sum(r_s) > best_sum:
                best_sum = np.sum(r_s)
                best_c = c_s
                best_r = r_s
                
    # Final strict repair
    centers = best_c.copy()
    radii = repair(centers, best_r.copy())
    return centers, radii, float(np.sum(radii))
