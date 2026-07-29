# sol_000247 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000214 (state 097281dc) state=8aae4080 sum of radii=2.606518 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_IND = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

def build_lp_matrices(n):
    """Pre-construct the sparse structure of the LP constraint matrix."""
    A = np.zeros((NUM_PAIRS + 4 * n, n))
    pairs = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            pairs.append((i, j))
            k += 1
    for i in range(n):
        base = NUM_PAIRS + 4 * i
        A[base, i] = 1.0
        A[base + 1, i] = 1.0
        A[base + 2, i] = 1.0
        A[base + 3, i] = 1.0
    return A, pairs

A_LP, LP_PAIRS = build_lp_matrices(N)

def solve_lp_radii(centers):
    """Solves LP for maximal radii given fixed centers and returns radii, sum, and duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
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
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if res.success:
            try:
                duals = res.ineqlin.marginals
            except AttributeError:
                duals = np.zeros_like(b)
            return res.x, np.sum(res.x), duals
    except Exception:
        pass
    
    # Fallback to simple distance-based radii if LP fails
    radii = np.minimum(ub, 0.5 * np.min(dists, axis=1))
    return radii, np.sum(radii), np.zeros_like(b)

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

def gradient_ascent(centers0, max_iter=4000, init_step=0.008):
    """Runs gradient ascent on centers to maximize sum of radii."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    patience = 0
    
    for k in range(max_iter):
        radii, curr_sum, duals = solve_lp_radii(centers)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            patience = 0
        else:
            patience += 1
            if patience > 50:
                step *= 0.85
            
        if step < 1e-12:
            break
            
        grad = compute_grad(centers, duals)
        gn = np.linalg.norm(grad)
        if gn < 1e-12:
            break
            
        centers += step * (grad / gn)
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
        # Periodic jitter to escape flat regions/local minima
        if k % 300 == 0 and k > 0:
            centers += np.random.normal(0, step * 0.15, centers.shape)
            centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
            
    return best_centers, best_sum

def slsqp_obj(v):
    """Objective: minimize negative sum of radii."""
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
    dx = c[TRIU_IND[0], 0] - c[TRIU_IND[1], 0]
    dy = c[TRIU_IND[0], 1] - c[TRIU_IND[1], 1]
    dr = r[TRIU_IND[0]] + r[TRIU_IND[1]]
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

def coord_obj(x, centers, idx):
    """Objective for coordinate-wise optimization: maximize LP sum for a single circle."""
    temp = centers.copy()
    temp[idx] = np.clip(x, 1e-5, 1.0 - 1e-5)
    _, s, _ = solve_lp_radii(temp)
    return -s

def coordinate_optimize(centers, cycles=3):
    """Optimizes each circle's position independently while fixing others."""
    c = centers.copy()
    for _ in range(cycles):
        for i in range(N):
            try:
                res = minimize(coord_obj, c[i], args=(c, i), method='Powell',
                               options={'maxiter': 400, 'xtol': 1e-8})
                if res.success:
                    c[i] = np.clip(res.x, 1e-5, 1.0 - 1e-5)
            except Exception:
                pass
    return c

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
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

def generate_inits(rng, n_inits=20):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5], [5, 7, 5, 5, 4]
    ]
    for pat in patterns:
        for r0 in [0.09, 0.095, 0.10, 0.105]:
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
            
    # Force-directed random starts to ensure good initial spacing
    for _ in range(6):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        f = (0.22 - d) / d * 0.015
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits[:n_inits]

def run_packing() -> tuple:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = generate_inits(rng, 20)
    
    # Phase 1: Gradient Ascent from diverse starts
    for c_init in inits:
        c_ga, s_ga = gradient_ascent(c_init, max_iter=4500, init_step=0.009)
        if s_ga > best_sum:
            best_sum = s_ga
            best_c = c_ga
            r_lp, _, _ = solve_lp_radii(best_c)
            best_r = r_lp
            
        # Aggressive perturbation to escape local minima
        c_pert = best_c + rng.normal(0, 0.02, best_c.shape)
        c_pert = np.clip(c_pert, 0.05, 0.95)
        c_ga2, s_ga2 = gradient_ascent(c_pert, max_iter=3000, init_step=0.007)
        if s_ga2 > best_sum:
            best_sum = s_ga2
            best_c = c_ga2
            r_lp, _, _ = solve_lp_radii(best_c)
            best_r = r_lp

    if best_c is not None:
        # Phase 2: Coordinate-wise refinement on the best configuration
        best_c = coordinate_optimize(best_c, cycles=3)
        r_lp, s_lp, _ = solve_lp_radii(best_c)
        best_r = r_lp
        best_sum = s_lp
        
        # Phase 3: Basin-hopping style jitter around the best
        for _ in range(10):
            c_jit = best_c + rng.normal(0, 0.008, best_c.shape)
            c_jit = np.clip(c_jit, 0.02, 0.98)
            c_opt, s_opt = gradient_ascent(c_jit, max_iter=2500, init_step=0.006)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt
                r_lp, _, _ = solve_lp_radii(best_c)
                best_r = r_lp

    # Phase 4: Joint SLSQP Polish
    if best_c is not None:
        ub = np.minimum(np.minimum(best_c[:, 0], 1.0 - best_c[:, 0]),
                        np.minimum(best_c[:, 1], 1.0 - best_c[:, 1]))
        dists = np.linalg.norm(best_c[:, None, :] - best_c[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(ub, rp) * 0.90
        
        c_polish, r_polish, s_polish = slsqp_optimize(best_c, r_init, maxiter=10000)
        if s_polish > best_sum:
            best_sum = s_polish
            best_c = c_polish
            best_r = r_polish
            
    # Phase 5: Final LP solve & Strict Repair
    r_final, s_final, _ = solve_lp_radii(best_c)
    if s_final > best_sum:
        best_sum = s_final
        best_r = r_final
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
