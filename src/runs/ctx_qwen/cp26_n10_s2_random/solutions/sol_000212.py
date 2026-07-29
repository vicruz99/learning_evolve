# sol_000212 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000169 (state 623e904f) state=3026fee6 sum of radii=2.624008 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)
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
    """Solves LP to maximize sum of radii for fixed centers. Returns radii and dual multipliers."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
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
            duals = np.zeros(A_LP.shape[0])
        return res.x, duals
    return np.zeros(N), np.zeros(A_LP.shape[0])

def get_gradient(centers):
    """Computes optimal radii, their sum, and the gradient w.r.t centers using LP duals."""
    radii, duals = solve_lp(centers)
    s = np.sum(radii)
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
        
    return radii, s, grad

def gradient_ascent(c0, max_iter=1500, step_init=0.005, rng=None):
    """Gradient ascent on centers to maximize sum of radii."""
    c = c0.copy()
    best_c = c.copy()
    best_s = -1.0
    step = step_init
    
    _, s, _ = get_gradient(c)
    best_s = s
    
    for _ in range(max_iter):
        radii, s, grad = get_gradient(c)
        if s > best_s:
            best_s = s
            best_c = c.copy()
            
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
            
        c_new = c + step * grad / g_norm
        c_new = np.clip(c_new, 0.005, 0.995)
        
        _, s_new, _ = get_gradient(c_new)
        if s_new > s + 1e-12:
            c = c_new
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.82
            
        if step < 1e-8:
            break
            
    return best_c, best_s

def slsqp_obj(v):
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """SLSQP constraints: boundary and pairwise non-overlap (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    i, j = IDX_I, IDX_J
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def polish(c_init, rng=None):
    """Refines centers and radii jointly using SLSQP."""
    radii, s, _ = get_gradient(c_init)
    v0 = np.concatenate([c_init.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 4000, 'ftol': 1e-13})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            c_opt = res.x[:2*N].reshape(N, 2)
            r_opt = res.x[2*N:]
            return c_opt, r_opt, np.sum(r_opt)
    except Exception:
        pass
    return c_init, radii, s

def generate_inits(rng):
    """Generates diverse initial configurations."""
    inits = []
    # Hexagonal patterns with varying row counts
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], [5,5,6,5,5], [5,4,6,6,5]]
    for pat in pats:
        for r0 in [0.095, 0.100, 0.105, 0.108]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                shift = r0 if ri % 2 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            inits.append(c)
            
    # Force-repelled random starts to find non-lattice optima
    for _ in range(12):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i]-c[j])
                    if d < 0.22 and d > 1e-5:
                        push = (0.22 - d) * 0.025
                        f[i] += (c[i]-c[j])/d * push
                        f[j] -= (c[i]-c[j])/d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(60):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c, best_r, best_s = None, None, -1.0
    
    inits = generate_inits(rng)
    
    # Phase 1: Gradient Ascent from diverse starts
    for c0 in inits:
        c_opt, s_opt = gradient_ascent(c0, rng=rng)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: SLSQP Polish & Basin-Hopping Perturbations
    if best_c is not None:
        c_p, r_p, s_p = polish(best_c, rng)
        if s_p > best_s:
            best_s = s_p
            best_c, best_r = c_p, r_p
            
        # Iterative perturbation to escape local minima
        for step_idx in range(20):
            noise_scale = 0.008 * (0.9 ** step_idx)
            c_jit = best_c + rng.normal(0, noise_scale, best_c.shape)
            c_jit = np.clip(c_jit, 0.05, 0.95)
            
            # Shrink radii slightly to ensure feasible restart for gradient ascent
            c_jit, s_jit = gradient_ascent(c_jit, max_iter=800, step_init=0.004, rng=rng)
            if s_jit > best_s:
                best_s = s_jit
                best_c = c_jit.copy()
                
            c_p2, r_p2, s_p2 = polish(c_jit, rng)
            if s_p2 > best_s:
                best_s = s_p2
                best_c, best_r = c_p2, r_p2
                
    # Final LP solve to match radii exactly to best centers
    if best_c is not None:
        if best_r is None:
            best_r, best_s, _ = get_gradient(best_c)
        else:
            _, s_final, _ = get_gradient(best_c)
            if s_final > best_s:
                best_r, best_s = None, s_final
                
    if best_r is None:
        best_r, best_s, _ = get_gradient(best_c)
        
    return best_c, repair(best_c, best_r), float(best_s)
