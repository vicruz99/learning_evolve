# sol_000270 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000237 (state 963256f0) state=03a03856 sum of radii=2.621304 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute LP constraint structure for pairwise distances: r_i + r_j <= dist(i, j)
_PAIR_IDX = [(i, j) for i in range(N) for j in range(i + 1, N)]
_NUM_PAIRS = len(_PAIR_IDX)

A_LP_pair = np.zeros((_NUM_PAIRS, N))
for k, (i, j) in enumerate(_PAIR_IDX):
    A_LP_pair[k, i] = 1.0
    A_LP_pair[k, j] = 1.0

def solve_lp_and_grad(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, sum_radii, and exact gradient w.r.t centers from active duals.
    """
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    
    # Upper bounds from square boundaries: r_i <= min(x, 1-x, y, 1-y)
    ub = np.minimum(
        np.minimum(c[:, 0], 1.0 - c[:, 0]),
        np.minimum(c[:, 1], 1.0 - c[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise Euclidean distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 0.0)
    
    b_pair = dists[np.triu_indices(N, k=1)]
    
    # LP: max sum(r) <=> min -sum(r)
    res = linprog(-np.ones(N), A_ub=A_LP_pair, b_ub=b_pair, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    
    # Extract dual marginals for pairwise constraints robustly
    duals_pair = np.zeros(_NUM_PAIRS)
    if hasattr(res, 'marginals') and res.marginals is not None:
        if hasattr(res.marginals, 'ineqlin'):
            duals_pair = res.marginals.ineqlin
        elif hasattr(res.marginals, 'ineq'):
            duals_pair = res.marginals.ineq
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals_pair = res.ineqlin.marginals
        
    grad = np.zeros_like(c)
    
    # Compute exact gradient from active pairwise repulsion constraints
    for k, (i, j) in enumerate(_PAIR_IDX):
        mu = duals_pair[k]
        if mu > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
                
    return radii, np.sum(radii), grad

def obj_lp(x):
    """Objective for center optimization: minimize negative sum of radii."""
    return -solve_lp_and_grad(x.reshape(N, 2))[1]

def grad_lp(x):
    """Exact gradient for center optimization."""
    return -solve_lp_and_grad(x.reshape(N, 2))[2].flatten()

def optimize_centers_lbps(c0):
    """Optimize circle centers using L-BFGS-B with exact LP gradient."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(obj_lp, c0.flatten(), jac=grad_lp, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2)
    except Exception:
        return c0

def joint_optimize_slsqp(centers, radii):
    """Joint constrained optimization on centers and radii for precise polishing."""
    n = centers.shape[0]
    v0 = np.concatenate([centers.flatten(), radii])
    
    def obj(v):
        return -np.sum(v[2 * n:])
        
    def cons(v):
        cc = v[:2 * n].reshape(n, 2)
        rr = v[2 * n:]
        c_list = [
            cc[:, 0] - rr,
            1.0 - cc[:, 0] - rr,
            cc[:, 1] - rr,
            1.0 - cc[:, 1] - rr
        ]
        idx = np.triu_indices(n, 1)
        d = np.linalg.norm(cc[idx[0]] - cc[idx[1]], axis=1)
        c_list.append(d - (rr[idx[0]] + rr[idx[1]]))
        return np.concatenate(c_list)
        
    bounds_j = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds_j,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-9:
            return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]
    except Exception:
        pass
    return centers, radii

def repair_packing(centers, radii):
    """Deterministically shrinks radii to guarantee strict validation compliance."""
    radii = radii.copy()
    n = centers.shape[0]
    
    for _ in range(80):
        changed = False
        # Boundary clamp
        for i in range(n):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
                
        # Pairwise overlap resolution
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    return np.maximum(radii, 0.0)

def generate_starts(rng):
    """Generates diverse, high-quality initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 4, 6, 5, 5], [5, 5, 6, 5, 5]]
    for p in pats:
        for r0 in [0.088, 0.095, 0.102, 0.108]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(p):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            while len(c) < N:
                c.append(rng.uniform(0.15, 0.85, 2))
            c = np.array(c[:N]) + rng.normal(0, 0.003, (N, 2))
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # 2. Perturbed 5x5 grid + 1 (26 circles naturally fit close to square grid)
    gs = np.linspace(0.12, 0.88, 5)
    cx, cy = np.meshgrid(gs, gs)
    g = np.column_stack([cx.flatten(), cy.flatten()])
    for i in range(8):
        c = g.copy()
        c = np.vstack([c, rng.uniform(0.2, 0.8, 2)])
        c += rng.normal(0, 0.015, (N, 2))
        c = np.clip(c, 0.08, 0.92)
        starts.append(c)
        
    # 3. Force-directed repulsion starts
    for _ in range(6):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(300):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(c[i] - c[j])
                    if 0.0 < d < 0.14:
                        push = (0.14 - d) * 0.02 / (d + 1e-6)
                        vec = (c[i] - c[j]) / d
                        f[i] += vec * push
                        f[j] -= vec * push
            c += f
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization on centers
    for c0 in starts:
        c_opt = optimize_centers_lbps(c0)
        r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    # Phase 2: Joint SLSQP polish on best configuration
    if best_c is not None:
        c_pol, r_pol = joint_optimize_slsqp(best_c, best_r)
        s_pol = np.sum(r_pol)
        if s_pol > best_s:
            best_s = s_pol
            best_c = c_pol
            best_r = r_pol
            
    # Phase 3: Simulated Annealing / Basin Hopping to escape local optima
    if best_c is not None:
        c_curr = best_c.copy()
        r_curr = best_r.copy()
        s_curr = best_s
        T = 0.006
        
        for step in range(900):
            T *= 0.9965
            c_new = c_curr + rng.normal(0, T, c_curr.shape)
            c_new = np.clip(c_new, 0.02, 0.98)
            
            r_new, s_new, _ = solve_lp_and_grad(c_new)
            
            # Metropolis criterion
            delta = s_new - s_curr
            if delta > 0 or rng.random() < np.exp(delta / max(T * 12.0, 1e-8)):
                c_curr = c_new
                r_curr = r_new
                s_curr = s_new
                if s_curr > best_s:
                    best_s = s_curr
                    best_c = c_curr.copy()
                    best_r = r_curr.copy()
                    
            # Periodic polishing during cooling
            if step % 120 == 0 and step > 0:
                c_p, r_p = joint_optimize_slsqp(c_curr, r_curr)
                s_p = np.sum(r_p)
                if s_p > best_s:
                    best_s = s_p
                    best_c = c_p
                    best_r = r_p
                    c_curr, r_curr, s_curr = c_p, r_p, s_p
                    
    # Fallback safety net
    if best_c is None:
        best_c = rng.uniform(0.1, 0.9, (N, 2))
        best_r = np.full(N, 0.05)
        
    # Phase 4: Strict numerical repair to guarantee validation passes
    best_r = repair_packing(best_c, best_r)
    final_sum = float(np.sum(best_r))
    
    return best_c, best_r, final_sum
