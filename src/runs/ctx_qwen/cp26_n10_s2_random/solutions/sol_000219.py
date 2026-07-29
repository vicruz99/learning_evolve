# sol_000219 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000168 (state 79899e79) state=d5cd93cb sum of radii=0.136178 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRI_U = np.triu_indices(N, k=1)
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N

# Precompute LP matrix structure globally
A_LP = np.zeros((NUM_PAIRS + NUM_BOUND, N))
PAIR_IDX = []
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        PAIR_IDX.append((i, j))
        k += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp(centers):
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in PAIR_IDX:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        return res.x, np.sum(res.x), res.ineqlin.marginals
    return np.zeros(N), 0.0, np.zeros_like(b)

def compute_grad(centers, duals):
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    idx = 0
    for i, j in PAIR_IDX:
        mu = duals[idx]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(n):
        mu_L = duals[bound_start + 4*i]
        mu_R = duals[bound_start + 4*i + 1]
        mu_B = duals[bound_start + 4*i + 2]
        mu_T = duals[bound_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
    return grad

def lp_gradient_ascent(c0, max_iter=4000, init_step=0.008, rng=None):
    c = c0.copy()
    best_c = c.copy()
    best_sum = -1.0
    step = init_step
    no_improve = 0
    
    for k in range(max_iter):
        r, s, duals = solve_lp(c)
        if s > best_sum:
            best_sum = s
            best_c = c.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 150:
            step *= 0.6
        elif no_improve > 50:
            step *= 0.85
            
        if step < 1e-12:
            break
            
        grad = compute_grad(c, duals)
        gn = np.linalg.norm(grad)
        if gn < 1e-13:
            break
            
        c += step * (grad / gn)
        c = np.clip(c, 1e-5, 1.0 - 1e-5)
        
        if rng is not None and k % 300 == 0 and k > 0:
            noise = step * 0.25
            c += rng.normal(0, noise, c.shape)
            c = np.clip(c, 1e-5, 1.0 - 1e-5)
            
    return best_c, best_sum

def slsqp_obj(v):
    return -np.sum(v[2::3])

def slsqp_cons(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    dx = c[TRI_U[0], 0] - c[TRI_U[1], 0]
    dy = c[TRI_U[0], 1] - c[TRI_U[1], 1]
    dr = r[TRI_U[0]] + r[TRI_U[1]]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def run_slsqp(c_init, r_init, maxiter=2000):
    v0 = np.concatenate([c_init.flatten(), r_init])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': maxiter, 'ftol': 1e-14})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c_init, r_init, 0.0

def repair(centers, radii):
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def spread_points(centers, rng, iterations=150):
    for _ in range(iterations):
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(centers)
        min_dist = 0.15
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < min_dist and d > 1e-6:
                    f = (min_dist - d) / d * 0.01
                    forces[i] += f * diffs[i, j]
                    forces[j] -= f * diffs[i, j]
                    
        centers += forces
        centers = np.clip(centers, 0.05, 0.95)
    return centers

def generate_inits(rng):
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5], [4, 5, 6, 6, 5]
    ]
    for pat in patterns:
        for r0 in [0.088, 0.092, 0.096, 0.100, 0.104, 0.108]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            inits.append(np.array(c[:N]))
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c = spread_points(c, rng, iterations=150)
        inits.append(c)
        
    for _ in range(10):
        c = np.zeros((N, 2))
        c[0] = [0.12, 0.12]
        c[1] = [0.88, 0.12]
        c[2] = [0.12, 0.88]
        c[3] = [0.88, 0.88]
        c[4:] = rng.uniform(0.2, 0.8, (N-4, 2))
        c += rng.normal(0, 0.01, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = generate_inits(rng)
    
    # Phase 1: SLSQP on multiple starts
    for c_init in inits:
        ub = np.minimum(np.minimum(c_init[:, 0], 1.0 - c_init[:, 0]),
                        np.minimum(c_init[:, 1], 1.0 - c_init[:, 1]))
        dists = np.linalg.norm(c_init[:, None, :] - c_init[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(ub, rp) * 0.85
        
        c_opt, r_opt, s_opt = run_slsqp(c_init, r_init, maxiter=2000)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    # Fallback initialization if all SLSQP fails
    if best_c is None:
        best_c = rng.uniform(0.2, 0.8, (N, 2))
        r_lp, s_lp, _ = solve_lp(best_c)
        best_sum = s_lp
        best_r = r_lp

    # Phase 2: LP Gradient Ascent refinement on best
    if best_c is not None:
        for _ in range(5):
            c_ga, s_ga = lp_gradient_ascent(best_c, max_iter=4000, init_step=0.008, rng=rng)
            if s_ga > best_sum:
                best_sum = s_ga
                best_c = c_ga
                r_lp, _, _ = solve_lp(best_c)
                best_r = r_lp
                
        # Phase 3: Perturbation + SLSQP to escape local minima
        for _ in range(10):
            c_per = best_c + rng.normal(0, 0.004, best_c.shape)
            c_per = np.clip(c_per, 0.02, 0.98)
            r_per, _, _ = solve_lp(c_per)
            c_opt, r_opt, s_opt = run_slsqp(c_per, r_per, maxiter=2000)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt
                best_r = r_opt
                
        # Phase 4: Final LP polish
        r_final, s_final, _ = solve_lp(best_c)
        if s_final > best_sum:
            best_sum = s_final
            best_r = r_final
            
    # Strict numerical repair
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
