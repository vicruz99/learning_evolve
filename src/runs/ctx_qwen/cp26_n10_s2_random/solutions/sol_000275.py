# sol_000275 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000237 (state 963256f0) state=e5332036 sum of radii=2.598791 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute constant LP constraint matrix structure for pairwise distances
A_LP = np.zeros((N * (N - 1) // 2, N))
PAIR_INDICES = []
_lp_row = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[_lp_row, i] = 1.0
        A_LP[_lp_row, j] = 1.0
        PAIR_INDICES.append((i, j))
        _lp_row += 1

TRIU_INDICES = np.triu_indices(N, k=1)

def solve_lp_and_gradient(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, sum_radii, and gradient w.r.t centers.
    """
    n = centers.shape[0]
    
    # Upper bounds from boundary constraints
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-6)
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 0.0)
    
    b_ub = dists[TRIU_INDICES]
    
    # Solve LP: max sum(r) s.t. r_i + r_j <= dist(i, j), 0 <= r_i <= ub_i
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract dual marginals safely
    duals_ineq = np.zeros(A_LP.shape[0])
    if hasattr(res, 'marginals') and res.marginals is not None:
        if hasattr(res.marginals, 'ineqlin'):
            duals_ineq = res.marginals.ineqlin
        elif hasattr(res.marginals, 'ineq'):
            duals_ineq = res.marginals.ineq
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals_ineq = res.ineqlin.marginals
        
    # Compute gradient from active pairwise constraints
    dual_mat = np.zeros((n, n))
    idx = 0
    for i, j in PAIR_INDICES:
        val = duals_ineq[idx]
        if val > 1e-10:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                dual_mat[i, j] = val
                dual_mat[j, i] = -val
        idx += 1
        
    unit_diff = diff / np.maximum(dists, 1e-9)[..., np.newaxis]
    grad = np.einsum('ij,ijk->ik', dual_mat, unit_diff)
    
    # Add boundary gradient components if boundary constraints are active
    for i in range(n):
        if radii[i] > ub[i] - 1e-6:
            x, y = centers[i]
            # Check which boundary is tightest
            vals = [x, 1.0 - x, y, 1.0 - y]
            min_idx = np.argmin(vals)
            if min_idx == 0: grad[i, 0] += 1.0
            elif min_idx == 1: grad[i, 0] -= 1.0
            elif min_idx == 2: grad[i, 1] += 1.0
            else: grad[i, 1] -= 1.0
            
    return radii, np.sum(radii), grad

def obj_l_bfgs(x_flat):
    """Objective and gradient for L-BFGS-B."""
    c = x_flat.reshape(N, 2)
    _, s, g = solve_lp_and_gradient(c)
    return -s, -g.flatten()

def optimize_centers_l_bfgs(c0, max_iter=1500):
    """Optimizes centers using L-BFGS-B."""
    bounds = [(0.001, 0.999)] * (2 * N)
    try:
        res = minimize(obj_l_bfgs, c0.flatten(), method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14})
        c_opt = res.x.reshape(N, 2)
        _, s_opt, _ = solve_lp_and_gradient(c_opt)
        return c_opt, s_opt
    except Exception:
        return c0, -1.0

def generate_lattice(pattern, r_est, rng):
    """Generates a hexagonal lattice configuration."""
    c = []
    y = r_est
    for r_idx, cnt in enumerate(pattern):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
    while len(c) < N:
        c.append(rng.uniform(0.2, 0.8, 2))
    c = np.array(c[:N])
    c += rng.normal(0, 0.004, c.shape)
    c = np.clip(c, 0.05, 0.95)
    return c

def repair_packing(centers, radii):
    """Deterministically shrinks radii to guarantee strict validation compliance."""
    radii = radii.copy()
    n = centers.shape[0]
    
    for _ in range(60):
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

def polish_slsqp(centers, radii):
    """Joint SLSQP refinement for precise constraint satisfaction."""
    n = centers.shape[0]
    v0 = np.concatenate([centers.flatten(), radii])
    
    def obj_joint(v):
        return -np.sum(v[2 * n:])
        
    def cons_joint(v):
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
        
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons_joint(res.x)) >= -1e-9:
            return res.x[:2 * n].reshape(n, 2), res.x[2 * n:], np.sum(res.x[2 * n:])
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [6, 5, 5, 5, 5], [5, 6, 6, 5, 4],
        [6, 4, 5, 6, 5], [5, 5, 4, 6, 6], [5, 6, 4, 5, 6]
    ]
    
    # Phase 1: Lattice starts + L-BFGS-B
    for pat in patterns:
        for r_est in [0.09, 0.095, 0.10, 0.105, 0.11]:
            c0 = generate_lattice(pat, r_est, rng)
            c_opt, s_opt = optimize_centers_l_bfgs(c0)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                
    # Phase 2: Force-directed starts
    for _ in range(8):
        c0 = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(300):
            forces = np.zeros_like(c0)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c0[i] - c0[j]
                    dist = np.linalg.norm(d_vec)
                    if 0.0 < dist < 0.15:
                        push = (0.15 - dist) * 0.05 / (dist + 1e-6)
                        forces[i] += d_vec / dist * push
                        forces[j] -= d_vec / dist * push
            c0 += forces
            c0 = np.clip(c0, 0.05, 0.95)
        c_opt, s_opt = optimize_centers_l_bfgs(c0)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()

    # Phase 3: Iterative Growth & Re-optimize
    if best_centers is not None:
        for _ in range(6):
            r_curr, _, _ = solve_lp_and_gradient(best_centers)
            # Perturb and re-optimize to escape local basins
            c_pert = best_centers + rng.normal(0, 0.003, best_centers.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            c_opt, s_opt = optimize_centers_l_bfgs(c_pert)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                
    # Phase 4: Simulated Annealing on centers
    if best_centers is not None:
        c_curr = best_centers.copy()
        s_curr = best_sum
        T = 0.006
        for step in range(800):
            c_new = c_curr + rng.normal(0, T, c_curr.shape)
            c_new = np.clip(c_new, 0.02, 0.98)
            _, s_new, _ = solve_lp_and_gradient(c_new)
            
            if s_new > s_curr or rng.random() < np.exp((s_new - s_curr) / max(T * 8.0, 1e-8)):
                c_curr = c_new
                s_curr = s_new
                if s_curr > best_sum:
                    best_sum = s_curr
                    best_centers = c_curr.copy()
            T *= 0.995
            
    # Phase 5: Joint SLSQP Polish
    if best_centers is not None:
        r_init, _, _ = solve_lp_and_gradient(best_centers)
        c_pol, r_pol, s_pol = polish_slsqp(best_centers, r_init)
        if s_pol > best_sum:
            best_sum = s_pol
            best_centers = c_pol
            best_radii = r_pol
        else:
            best_radii, _, _ = solve_lp_and_gradient(best_centers)

    # Fallback safety
    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, (N, 2))
        best_radii = np.full(N, 0.05)
        best_sum = np.sum(best_radii)
    elif best_radii is None:
        best_radii, _, _ = solve_lp_and_gradient(best_centers)
        
    # Final strict repair
    best_radii = repair_packing(best_centers, best_radii)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
