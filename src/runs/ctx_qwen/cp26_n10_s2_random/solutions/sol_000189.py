# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000143 (state c665f9c9) state=c08080b7 sum of radii=2.604392 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def _get_lp_structure(n):
    """Pre-construct the sparse structure of the LP constraint matrix."""
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A_ub = np.zeros((num_pairs + num_bound, n))
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_indices.append((i, j))
            idx += 1
    for i in range(n):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1
    return A_ub, pair_indices

A_LP, PAIRS = _get_lp_structure(N)

def _solve_lp_and_grad(centers):
    """
    Solves LP for optimal radii given fixed centers and computes the gradient 
    of the sum of radii with respect to center positions using LP duals.
    """
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in PAIRS:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, -1.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract dual variables safely across scipy versions
    if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
        duals = res.ineqlin.marginals
    else:
        duals = np.zeros_like(b_ub)
        
    grad = np.zeros_like(centers)
    
    idx = 0
    for i, j in PAIRS:
        lam = duals[idx]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-12:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    boundary_start = len(PAIRS)
    for i in range(n):
        mu_L = duals[boundary_start + 4*i]
        mu_R = duals[boundary_start + 4*i + 1]
        mu_B = duals[boundary_start + 4*i + 2]
        mu_T = duals[boundary_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, np.sum(radii), grad

def _optimize_centers(c0, max_iter=2500, seed=0):
    """Runs gradient ascent on centers to maximize sum of radii."""
    rng = np.random.default_rng(seed)
    c = c0.copy()
    best_c = c.copy()
    best_sum = -1.0
    step = 0.008
    
    r, s, g = _solve_lp_and_grad(c)
    if r is None:
        return c0, -1.0
    best_sum = s
    
    for k in range(max_iter):
        r, s, g = _solve_lp_and_grad(c)
        if r is None:
            break
            
        if s > best_sum:
            best_sum = s
            best_c = c.copy()
            step = min(step * 1.12, 0.025)
        else:
            step *= 0.94
            
        gn = np.linalg.norm(g)
        if gn > 1e-9:
            c += step * g / gn
        else:
            step *= 0.5
            
        c = np.clip(c, 0.005, 0.995)
        
        # Periodic jitter to escape local minima
        if k > 0 and k % 150 == 0:
            noise_scale = 0.004 * np.exp(-k / 800.0)
            c += rng.normal(0, noise_scale, c.shape)
            c = np.clip(c, 0.02, 0.98)
            r, s, g = _solve_lp_and_grad(c)
            
        if step < 1e-8:
            break
            
    return best_c, best_sum

def _hex_init(pat, r0, rng):
    """Generates hexagonal lattice positions based on row counts."""
    c = []
    y = r0
    for r_idx, cnt in enumerate(pat):
        shift = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
    c = np.array(c[:N])
    c += rng.normal(0, 0.003, c.shape)
    c = np.clip(c, 0.05, 0.95)
    return c

def _force_init(rng):
    """Generates initial configuration via simple repulsive forces."""
    c = rng.uniform(0.2, 0.8, (N, 2))
    for _ in range(600):
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = c[i] - c[j]
                dist = np.linalg.norm(d_vec)
                if dist < 0.18 and dist > 1e-6:
                    push = (0.18 - dist) * 0.6
                    f = d_vec / dist * push
                    forces[i] += f
                    forces[j] -= f
        c += forces * 0.15
        c = np.clip(c, 0.1, 0.9)
    return c

def _repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def _slsqp_objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def _slsqp_constraints(v):
    """Boundary and pairwise non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
    idx_i, idx_j = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx_i] - c[idx_j], axis=1)
    con.append(d - (r[idx_i] + r[idx_j]))
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 6, 5, 5],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [5, 5, 5, 6, 5], [6, 4, 6, 5, 5], [5, 5, 4, 6, 6],
        [7, 5, 5, 5, 4], [5, 7, 5, 5, 4], [4, 5, 6, 5, 6]
    ]
    
    for pat in patterns:
        for r0 in [0.092, 0.098, 0.105, 0.112]:
            inits.append(_hex_init(pat, r0, rng))
            
    for _ in range(15):
        inits.append(_force_init(rng))
        
    for _ in range(10):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Phase 2: Gradient Ascent from multiple starts
    for i, c0 in enumerate(inits):
        c_opt, s_opt = _optimize_centers(c0, max_iter=2000, seed=i)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        return _hex_init([5,5,6,5,5], 0.1, rng), np.full(N, 0.1), 2.6

    r_lp, _, _ = _solve_lp_and_grad(best_c)
    if r_lp is not None:
        best_r = r_lp.copy()
        
    # Phase 3: SLSQP Joint Polish
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    v0 = np.concatenate([best_c.flatten(), best_r])
    
    for _ in range(4):
        try:
            res = minimize(_slsqp_objective, v0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': _slsqp_constraints},
                           options={'maxiter': 12000, 'ftol': 1e-14})
            if np.min(_slsqp_constraints(res.x)) >= -1e-7:
                s_new = np.sum(res.x[2*N:])
                if s_new > best_sum:
                    best_sum = s_new
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Strict numerical repair
    best_r = _repair(best_c, best_r)
    
    return best_c, best_r, float(np.sum(best_r))
