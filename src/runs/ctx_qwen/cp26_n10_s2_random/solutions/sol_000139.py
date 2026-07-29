# sol_000139 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000119 (state ab7c4e6b) state=e7b4b813 sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
# Precompute indices for pairwise constraints (i < j)
IDX_I, IDX_J = np.triu_indices(N, k=1)
PAIR_COUNT = len(IDX_I)

# Precompute the static inequality constraint matrix for LP: r_i + r_j <= dist_ij
A_LP = np.zeros((PAIR_COUNT, N))
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        idx += 1

def solve_lp_for_radii(centers):
    """
    Given fixed centers, solve LP to exactly maximize sum of radii.
    Constraints: r_i + r_j <= dist(i,j), 0 <= r_i <= dist_to_boundary
    """
    n = centers.shape[0]
    
    # Upper bound from boundaries
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    b_ub = dists[IDX_I, IDX_J]
    
    c_obj = -np.ones(n)
    bounds = [(0.0, u) for u in ub]
    
    try:
        res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_joint(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints_joint(v):
    """
    Computes all boundary and non-overlap constraints for SLSQP.
    Returns a 1D array where each element must be >= 0.
    """
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = []
    
    # Boundary constraints
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
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
    
    # 1. Hexagonal lattice patterns with varying row structures
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [6,4,6,5,5],
        [4,6,6,6,4], [5,4,6,6,5], [6,6,5,5,4], [5,5,6,5,5],
        [4,5,6,5,6], [5,5,5,6,5], [5,5,4,6,6], [6,5,5,5,5],
        [5,5,5,5,6], [6,6,4,5,5], [5,6,4,5,6], [6,5,5,6,4],
        [5,7,5,5,4], [5,5,7,5,4], [6,6,6,4,4], [5,5,5,7,4]
    ]
    
    for pat in patterns:
        for r_est in [0.09, 0.095, 0.10, 0.105, 0.11]:
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
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # 2. Force-relaxed random starts to encourage natural spacing
    for _ in range(40):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(60):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        f = (0.22 - d) * 0.6 / d
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces * 0.012
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_joint}
    
    # Phase 1: Multi-start constrained optimization + LP refinement
    for i, c_init in enumerate(starts):
        r_init = np.full(N, 0.06)
        v0 = np.concatenate([c_init.flatten(), r_init])
        
        try:
            res = minimize(objective_joint, v0, method='SLSQP', bounds=bounds_j,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-14})
            if hasattr(res, 'x'):
                c_curr = res.x[:2*N].reshape(N, 2)
                r_lp, s_lp = solve_lp_for_radii(c_curr)
                
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_curr.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # Fallback if optimization failed entirely
    if best_centers is None:
        c0 = rng.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum = solve_lp_for_radii(c0)
        best_centers = c0

    # Phase 2: Adaptive perturbation search to escape local minima
    if best_centers is not None:
        for step in range(40):
            noise = 0.005 * (0.82 ** step)
            c_pert = best_centers + rng.normal(0, noise, best_centers.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert = np.full(N, 0.08)
            v_pert = np.concatenate([c_pert.flatten(), r_pert])
            
            try:
                res = minimize(objective_joint, v_pert, method='SLSQP', bounds=bounds_j,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14})
                if hasattr(res, 'x'):
                    c_curr = res.x[:2*N].reshape(N, 2)
                    r_lp, s_lp = solve_lp_for_radii(c_curr)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_curr.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass

    # Phase 3: Strict numerical repair to guarantee validation passes
    for _ in range(50):
        changed = False
        # Fix boundary violations
        for i in range(N):
            mx = min(best_centers[i,0], 1.0-best_centers[i,0], 
                     best_centers[i,1], 1.0-best_centers[i,1])
            if best_radii[i] > mx - 1e-12:
                best_radii[i] = mx
                changed = True
        # Fix overlaps
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
