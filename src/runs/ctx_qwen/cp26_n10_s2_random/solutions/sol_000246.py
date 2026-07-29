# sol_000246 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000214 (state 097281dc) state=2bbead6c sum of radii=2.607214 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_IND = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
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

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii, sum, and dual multipliers."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    dx = centers[:, 0, None] - centers[None, :, 0]
    dy = centers[:, 1, None] - centers[None, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
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
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0, u) for u in ub], method='highs')
    if res.success:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except AttributeError:
            duals = np.zeros_like(b)
        return res.x, np.sum(res.x), duals
    return np.zeros(N), 0.0, np.zeros_like(b)

def get_gradient(centers, duals):
    """Computes exact gradient of sum of radii w.r.t centers using LP duals."""
    grad = np.zeros_like(centers)
    dx = centers[:, 0, None] - centers[None, :, 0]
    dy = centers[:, 1, None] - centers[None, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
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
        
    bound_start = NUM_PAIRS
    for i in range(N):
        mu_x0 = duals[bound_start + 4 * i]
        mu_x1 = duals[bound_start + 4 * i + 1]
        mu_y0 = duals[bound_start + 4 * i + 2]
        mu_y1 = duals[bound_start + 4 * i + 3]
        grad[i, 0] += mu_x0 - mu_x1
        grad[i, 1] += mu_y0 - mu_y1
    return grad

def gradient_ascent(c0, max_iter=7000, step_init=0.009, rng=None):
    """Gradient ascent on centers to maximize sum of radii with adaptive stepping."""
    c = c0.copy()
    best_c = c.copy()
    best_s = -1.0
    step = step_init
    
    _, s, duals = solve_lp(c)
    best_s = s
    patience = 0
    
    for it in range(max_iter):
        radii, s, duals = solve_lp(c)
        if s > best_s + 1e-12:
            best_s = s
            best_c = c.copy()
            patience = 0
        else:
            patience += 1
            
        # Adaptive step size control
        if patience > 80:
            step *= 0.75
        elif patience == 0:
            step = min(step * 1.06, 0.035)
            
        if step < 1e-11:
            break
            
        grad = get_gradient(c, duals)
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-12:
            break
            
        c_new = c + step * grad / g_norm
        c_new = np.clip(c_new, 1e-5, 1.0 - 1e-5)
        
        # Line search check
        _, s_new, _ = solve_lp(c_new)
        if s_new > s + 1e-12:
            c = c_new
        else:
            step *= 0.82
            
    return best_c, best_s

def slsqp_obj(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    i, j = TRIU_IND
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

SLSQP_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

def slsqp_optimize(c_init, r_init, maxiter=8000):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c_init.flatten(), r_init])
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=SLSQP_BOUNDS,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': maxiter, 'ftol': 1e-14})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c_init, r_init, 0.0

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], 
                     centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], 
                             centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d - 1e-11:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def generate_inits(rng, n_inits=50):
    """Generates diverse initial configurations."""
    inits = []
    
    # Hexagonal patterns with varying row counts
    pats = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], 
        [5,5,6,5,5], [5,4,6,6,5], [6,6,5,5,4], [5,7,5,5,4],
        [6,5,5,5,5], [5,5,5,6,5], [6,4,6,5,5], [4,5,6,5,6]
    ]
    for pat in pats:
        for r0 in [0.088, 0.094, 0.100, 0.106, 0.112]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                shift = r0 if ri % 2 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.0025), y + rng.normal(0, 0.0025)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            c = np.array(c[:N])
            c = np.clip(c, 0.04, 0.96)
            inits.append(c)
            
    # Force-repelled random starts to find non-lattice optima
    for _ in range(15):
        c = rng.uniform(0.12, 0.88, (N, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        push = (0.22 - d) * 0.02
                        f[i] += d_vec/d * push
                        f[j] -= d_vec/d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    # Corner-biased starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.09, 0.09], [0.91, 0.09], [0.09, 0.91], [0.91, 0.91]]
        for i, corner in enumerate(corners):
            c[i] = corner + rng.normal(0, 0.015, 2)
        inits.append(np.clip(c, 0.02, 0.98))
        
    return inits[:n_inits]

def run_packing() -> tuple:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_c, best_r, best_s = None, None, -1.0
    
    inits = generate_inits(rng, 55)
    
    # Phase 1: Multi-start Gradient Ascent
    for c0 in inits:
        c_opt, s_opt = gradient_ascent(c0, max_iter=6500, step_init=0.0085, rng=rng)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, _, _ = solve_lp(best_c)
    else:
        best_c = inits[0]
        best_r, best_s, _ = solve_lp(best_c)
        
    # Phase 2: Basin Hopping & Perturbation
    for step_idx in range(18):
        noise_scale = 0.012 * (0.85 ** (step_idx / 6.0))
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, 0.03, 0.97)
        
        c_opt, s_opt = gradient_ascent(c_pert, max_iter=4000, step_init=0.006, rng=rng)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp(best_c)
            
    # Phase 3: SLSQP Joint Polish
    r_lp, _, _ = solve_lp(best_c)
    c_slsqp, r_slsqp, s_slsqp = slsqp_optimize(best_c, r_lp, maxiter=9000)
    if s_slsqp > best_s:
        best_s = s_slsqp
        best_c = c_slsqp.copy()
        best_r = r_slsqp.copy()
        
    # Phase 4: Final LP solve & Strict Repair
    r_final, _, _ = solve_lp(best_c)
    radii = repair(best_c.copy(), r_final.copy())
    
    return best_c, radii, float(np.sum(radii))
