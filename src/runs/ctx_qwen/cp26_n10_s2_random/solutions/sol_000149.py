# sol_000149 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000139 (state e7b4b813) state=e9a338e7 sum of radii=2.503816 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)
PAIR_COUNT = len(IDX_I)

# Precompute static constraint matrix for LP: r_i + r_j <= dist_ij and boundary constraints
A_PAIRS = np.zeros((PAIR_COUNT, N))
for idx, (i, j) in enumerate(zip(IDX_I, IDX_J)):
    A_PAIRS[idx, i] = 1.0
    A_PAIRS[idx, j] = 1.0

A_BND = np.zeros((4 * N, N))
for i in range(N):
    A_BND[4*i, i] = 1.0
    A_BND[4*i+1, i] = 1.0
    A_BND[4*i+2, i] = 1.0
    A_BND[4*i+3, i] = 1.0

A_LP = np.vstack([A_PAIRS, A_BND])

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and returns gradient w.r.t centers."""
    n = centers.shape[0]
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    b_ub = np.concatenate([
        dists[IDX_I, IDX_J],
        centers[:, 0], 1.0 - centers[:, 0],
        centers[:, 1], 1.0 - centers[:, 1]
    ])
    
    c_obj = -np.ones(n)
    bounds = [(0.0, u) for u in ub]
    
    try:
        res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    total_sum = -res.fun
    # Marginals are shadow prices (non-negative for <= constraints when minimizing)
    marginals = np.maximum(res.ineqlin.marginals, 0.0)
    
    grad = np.zeros_like(centers)
    idx_pair = 0
    for i, j in zip(IDX_I, IDX_J):
        lam = marginals[idx_pair]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx_pair += 1
        
    idx_bnd = PAIR_COUNT
    for i in range(n):
        mu_x = marginals[idx_bnd + 4*i]
        mu_1x = marginals[idx_bnd + 4*i + 1]
        mu_y = marginals[idx_bnd + 4*i + 2]
        mu_1y = marginals[idx_bnd + 4*i + 3]
        grad[i, 0] += mu_x - mu_1x
        grad[i, 1] += mu_y - mu_1y
        
    return radii, total_sum, grad

def obj_and_grad_centers(c_flat):
    """Objective and gradient for center optimization (minimize negative sum of radii)."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, 1e-5, 1.0 - 1e-5)
    r, s, g = solve_lp_and_grad(c)
    return -s, -g.reshape(-1)

def obj_joint(v):
    """Objective for joint center+radius optimization."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint optimization."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = []
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    dx = c[IDX_I, 0] - c[IDX_J, 0]
    dy = c[IDX_I, 1] - c[IDX_J, 1]
    cons.append(np.hypot(dx, dy) - (r[IDX_I] + r[IDX_J]))
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    starts = []
    
    # 1. Hexagonal lattice patterns with varying structures
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [6,4,6,5,5],
        [4,6,6,6,4], [5,4,6,6,5], [6,6,5,5,4], [5,5,6,5,5],
        [4,5,6,5,6], [5,5,5,6,5], [5,5,4,6,6], [6,5,5,5,5],
        [5,5,5,5,6], [6,6,4,5,5], [5,6,4,5,6], [6,5,5,6,4],
        [5,7,5,5,4], [5,5,7,5,4], [6,6,6,4,4], [5,5,5,7,4]
    ]
    for pat in patterns:
        for r_est in [0.092, 0.098, 0.104, 0.110]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.02, 0.98)
            starts.append(c)
            
    # 2. Force-relaxed random starts to encourage natural spacing
    for _ in range(30):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(50):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.20 and d > 1e-6:
                        f = (0.20 - d) * 0.5 / d
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces * 0.01
        c = np.clip(c, 0.02, 0.98)
        starts.append(c)
        
    bounds_c = [(0.0, 1.0)] * (2 * N)
    bounds_j = bounds_c + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': cons_joint}
    
    # Phase 1: L-BFGS-B optimization on centers using exact LP gradients
    for c_init in starts:
        try:
            res = minimize(obj_and_grad_centers, c_init.flatten(), method='L-BFGS-B',
                           jac=True, bounds=bounds_c, options={'maxiter': 500, 'ftol': 1e-12})
            c_curr = np.clip(res.x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
            r_lp, s_lp, _ = solve_lp_and_grad(c_curr)
            
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_curr.copy()
                best_radii = r_lp.copy()
        except Exception:
            pass
            
    if best_centers is None:
        c0 = rng.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum, _ = solve_lp_and_grad(c0)
        best_centers = c0

    # Phase 2: Joint SLSQP polish to fine-tune contacts
    v0 = np.concatenate([best_centers.flatten(), best_radii])
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_j,
                       constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14})
        if hasattr(res, 'x'):
            c_curr = res.x[:2*N].reshape(N, 2)
            r_lp, s_lp, _ = solve_lp_and_grad(c_curr)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_curr.copy()
                best_radii = r_lp.copy()
    except Exception:
        pass
        
    # Phase 3: Adaptive perturbation search to escape local minima
    for step in range(30):
        noise = 0.004 * (0.85 ** step)
        c_pert = best_centers + rng.normal(0, noise, best_centers.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        try:
            res = minimize(obj_and_grad_centers, c_pert.flatten(), method='L-BFGS-B',
                           jac=True, bounds=bounds_c, options={'maxiter': 300, 'ftol': 1e-12})
            c_curr = np.clip(res.x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
            r_lp, s_lp, _ = solve_lp_and_grad(c_curr)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_curr.copy()
                best_radii = r_lp.copy()
        except Exception:
            pass
            
    # Phase 4: Final exact LP and strict numerical repair
    best_radii, best_sum, _ = solve_lp_and_grad(best_centers)
    
    for _ in range(50):
        changed = False
        for i in range(N):
            mx = min(best_centers[i,0], 1.0-best_centers[i,0], 
                     best_centers[i,1], 1.0-best_centers[i,1])
            if best_radii[i] > mx - 1e-12:
                best_radii[i] = mx
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], 
                             best_centers[i,1]-best_centers[j,1])
                if d < best_radii[i] + best_radii[j] - 1e-12:
                    shrink = (best_radii[i] + best_radii[j] - d) / 2.0 + 1e-9
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
