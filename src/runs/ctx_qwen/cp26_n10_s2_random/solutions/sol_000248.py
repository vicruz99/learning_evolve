# sol_000248 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000214 (state 097281dc) state=99d5cc58 sum of radii=2.556650 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_IND = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant structure of the LP constraint matrix
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
LP_PAIRS = []
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        LP_PAIRS.append((i, j))
        k += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_radii(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and dual multipliers."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in LP_PAIRS:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        try:
            duals = res.ineqlin.marginals
        except (AttributeError, TypeError):
            duals = np.zeros_like(b)
        return res.x, np.sum(res.x), duals
    return np.zeros(n), 0.0, np.zeros_like(b)

def compute_grad(centers, duals):
    """Computes exact gradient of sum of radii w.r.t centers using LP duals."""
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    idx = 0
    for i, j in LP_PAIRS:
        mu = duals[idx]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = NUM_PAIRS
    for i in range(n):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
    return grad

def gradient_ascent(centers, max_iter=2000, init_step=0.005, rng=None):
    """Runs gradient ascent on centers to maximize sum of radii."""
    c = centers.copy()
    best_c = c.copy()
    best_sum = -1.0
    step = init_step
    if rng is None:
        rng = np.random.default_rng(0)
        
    _, s, duals = solve_lp_radii(c)
    best_sum = s
    
    for k in range(max_iter):
        grad = compute_grad(c, duals)
        gn = np.linalg.norm(grad)
        if gn < 1e-10:
            break
            
        c_new = c + step * (grad / gn)
        c_new = np.clip(c_new, 1e-5, 1.0 - 1e-5)
        
        r_new, s_new, duals_new = solve_lp_radii(c_new)
        if s_new > best_sum:
            best_sum = s_new
            best_c = c_new.copy()
            step = min(step * 1.08, 0.03)
        else:
            step *= 0.85
            
        c = c_new
        duals = duals_new
        
        if step < 1e-10:
            break
            
        # Occasional jitter to break symmetry/plateaus
        if k > 0 and k % 150 == 0:
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 1e-5, 1.0 - 1e-5)
            _, s, duals = solve_lp_radii(c)
            if s > best_sum:
                best_sum = s
                best_c = c.copy()
                
    return best_c, best_sum

def simulated_annealing(centers, T_init=0.006, T_min=1e-6, cooling=0.998, steps=2000, rng=None):
    """Simulated annealing on centers using LP objective."""
    c = centers.copy()
    _, best_s, _ = solve_lp_radii(c)
    best_c = c.copy()
    curr_s = best_s
    T = T_init
    if rng is None:
        rng = np.random.default_rng(0)
        
    for k in range(steps):
        sigma = max(0.0015, T * 0.4)
        c_new = c + rng.normal(0, sigma, c.shape)
        c_new = np.clip(c_new, 1e-4, 1.0 - 1e-4)
        
        _, s_new, _ = solve_lp_radii(c_new)
        
        delta = s_new - curr_s
        if delta > 0 or rng.random() < np.exp(delta / T):
            c = c_new
            curr_s = s_new
            if curr_s > best_s:
                best_s = curr_s
                best_c = c.copy()
        T *= cooling
        if T < T_min:
            break
            
    return best_c, best_s

def slsqp_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    dx = c[TRIU_IND[0], 0] - c[TRIU_IND[1], 0]
    dy = c[TRIU_IND[0], 1] - c[TRIU_IND[1], 1]
    dr = r[TRIU_IND[0]] + r[TRIU_IND[1]]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

SLSQP_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

def slsqp_optimize(c_init, r_init, maxiter=6000):
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
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def generate_inits(rng, n_inits=40):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]
    ]
    for pat in patterns:
        for r0 in [0.090, 0.095, 0.100, 0.105]:
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
            
    # Corner & edge biased starts with repulsion
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        corners = np.array([[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]])
        c[:4] = corners + rng.normal(0, 0.015, (4, 2))
        for _ in range(300):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.2 and d > 1e-6:
                        f = (0.2 - d) / d * 0.01
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces
            c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    return inits[:n_inits]

def run_packing() -> tuple:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = generate_inits(rng, 35)
    
    # Phase 1: Gradient Ascent from diverse starts
    for c_init in inits:
        c_opt, s_opt = gradient_ascent(c_init, max_iter=2500, init_step=0.006, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = inits[0]
        
    r_lp, best_sum, _ = solve_lp_radii(best_c)
    best_r = r_lp
    
    # Phase 2: Simulated Annealing & Gradient Ascent cycles
    for cycle in range(6):
        c_sa, s_sa = simulated_annealing(best_c, T_init=0.008 * (0.85**cycle), T_min=1e-5, steps=1800, rng=rng)
        if s_sa > best_sum:
            best_sum = s_sa
            best_c = c_sa.copy()
            
        c_ga, s_ga = gradient_ascent(best_c, max_iter=2000, init_step=0.005, rng=rng)
        if s_ga > best_sum:
            best_sum = s_ga
            best_c = c_ga.copy()
            
        r_lp, _, _ = solve_lp_radii(best_c)
        best_r = r_lp
        
    # Phase 3: SLSQP Joint Polish
    c_slsqp, r_slsqp, s_slsqp = slsqp_optimize(best_c, best_r, maxiter=8000)
    if s_slsqp > best_sum:
        best_sum = s_slsqp
        best_c = c_slsqp
        best_r = r_slsqp
        
    # Final LP solve to ensure radii exactly match centers
    r_final, s_final, _ = solve_lp_radii(best_c)
    if s_final > best_sum:
        best_sum = s_final
        best_r = r_final
        
    # Strict numerical repair for validation
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
