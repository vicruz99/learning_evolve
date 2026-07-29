# sol_000273 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000237 (state 963256f0) state=bd65c07a sum of radii=2.624008 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute constant LP constraint matrix structure for pairwise distances
# Constraint: r_i + r_j <= dist(i, j)
A_LP = np.zeros((N * (N - 1) // 2, N))
PAIR_INDICES = []
_lp_row = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[_lp_row, i] = 1.0
        A_LP[_lp_row, j] = 1.0
        PAIR_INDICES.append((i, j))
        _lp_row += 1

def solve_lp_and_gradient(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, sum_radii, and gradient w.r.t centers using dual marginals.
    """
    n = centers.shape[0]
    
    # Upper bounds from boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    b_ub = dists[np.triu_indices(n, k=1)]
    
    # Solve LP: max sum(r) s.t. A r <= b, 0 <= r <= ub
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract dual marginals safely
    duals_pair = np.zeros(A_LP.shape[0])
    duals_ub = np.zeros(n)
    
    marg = getattr(res, 'marginals', None)
    if marg is not None:
        if hasattr(marg, 'ineqlin'):
            duals_pair = marg.ineqlin
        if hasattr(marg, 'upper'):
            duals_ub = -marg.upper  # Negate because linprog minimizes -sum(r)
            
    # Compute gradient from active pairwise constraints
    grad = np.zeros_like(centers)
    idx = 0
    for i, j in PAIR_INDICES:
        lam = duals_pair[idx]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    # Compute gradient from active boundary constraints
    for i in range(n):
        mu = duals_ub[i]
        if mu > 1e-9:
            x, y = centers[i]
            if abs(ub[i] - x) < 1e-7: grad[i, 0] += mu
            elif abs(ub[i] - (1.0 - x)) < 1e-7: grad[i, 0] -= mu
            elif abs(ub[i] - y) < 1e-7: grad[i, 1] += mu
            elif abs(ub[i] - (1.0 - y)) < 1e-7: grad[i, 1] -= mu
            
    return radii, np.sum(radii), grad

def generate_inits(rng):
    """Generates diverse starting configurations."""
    configs = []
    
    # 1. Hexagonal lattice patterns with varying densities
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]
    ]
    for pat in patterns:
        for r0 in [0.095, 0.102, 0.108]:
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
            while len(c) < N:
                c.append(rng.uniform(0.1, 0.9, 2))
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            configs.append(c)
            
    # 2. Corner & Edge biased starts
    for _ in range(6):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[4:8] = [[0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        c += rng.normal(0, 0.01, c.shape)
        c = np.clip(c, 0.02, 0.98)
        configs.append(c)
        
    # 3. Force-directed repulsion starts
    for _ in range(6):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if 0.0 < dist < 0.12:
                        push = (0.12 - dist) * 0.04 / (dist + 1e-6)
                        forces[i] += d_vec / dist * push
                        forces[j] -= d_vec / dist * push
            c += forces
            c = np.clip(c, 0.05, 0.95)
        configs.append(c)
        
    return configs

def gradient_ascent(c0, rng, max_iter):
    """Adaptive gradient ascent with automatic basin hopping."""
    c = c0.copy()
    step = 0.006
    best_sum = -1.0
    best_c = c.copy()
    no_improve = 0
    
    for k in range(max_iter):
        radii, curr_sum, grad = solve_lp_and_gradient(c)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_c = c.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        gn = np.linalg.norm(grad)
        if gn < 1e-12:
            break
            
        c_new = c + step * grad / gn
        c_new = np.clip(c_new, 0.005, 0.995)
        
        _, sum_new, _ = solve_lp_and_gradient(c_new)
        
        if sum_new > curr_sum - 1e-12:
            c = c_new
            step = min(step * 1.05, 0.03)
        else:
            step *= 0.5
            
        # Automatic basin hopping if stuck
        if no_improve > 100:
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.02, 0.98)
            no_improve = 0
            step = 0.005
            
        if step < 1e-8:
            break
            
    return best_c, best_sum

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared distances)."""
    cc = v[:2 * N].reshape(N, 2)
    rr = v[2 * N:]
    c_list = [
        cc[:, 0] - rr,
        1.0 - cc[:, 0] - rr,
        cc[:, 1] - rr,
        1.0 - cc[:, 1] - rr
    ]
    idx = np.triu_indices(N, 1)
    dx = cc[idx[0], 0] - cc[idx[1], 0]
    dy = cc[idx[0], 1] - cc[idx[1], 1]
    dr = rr[idx[0]] + rr[idx[1]]
    c_list.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c_list)

def polish_slsqp(centers, radii):
    """Joint SLSQP refinement for precise constraint satisfaction."""
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-9:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], np.sum(res.x[2 * N:])
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def repair_packing(centers, radii):
    """Deterministically shrinks radii to guarantee strict validation compliance."""
    radii = radii.copy()
    n = centers.shape[0]
    
    for _ in range(50):
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

def get_score(item):
    """Helper for sorting without lambdas."""
    return item[1]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Phase 1: Diverse starts & Gradient Ascent
    configs = generate_inits(rng)
    candidates = []
    
    for c0 in configs:
        c_opt, s_opt = gradient_ascent(c0, rng, max_iter=1500)
        candidates.append((c_opt, s_opt))
        
    # Sort by performance and keep top candidates for intensive polishing
    candidates.sort(key=get_score, reverse=True)
    
    # Phase 2: Polish top candidates with SLSQP
    for c_opt, s_opt in candidates[:6]:
        radii, _, _ = solve_lp_and_gradient(c_opt)
        c_pol, r_pol, s_pol = polish_slsqp(c_opt, radii)
        if s_pol > best_sum:
            best_sum = s_pol
            best_centers = c_pol
            best_radii = r_pol
            
    # Phase 3: Iterative perturbation to escape local minima
    if best_centers is not None:
        c_curr = best_centers.copy()
        r_curr = best_radii.copy()
        s_curr = best_sum
        
        for trial in range(15):
            noise = 0.005 * (0.9 ** trial)
            c_pert = c_curr + rng.normal(0, noise, c_curr.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            c_opt, s_opt = gradient_ascent(c_pert, rng, max_iter=1000)
            if s_opt > best_sum:
                radii, _, _ = solve_lp_and_gradient(c_opt)
                c_pol, r_pol, s_pol = polish_slsqp(c_opt, radii)
                if s_pol > best_sum:
                    best_sum = s_pol
                    best_centers = c_pol
                    best_radii = r_pol
                    c_curr, r_curr, s_curr = c_pol, r_pol, s_pol
                    
    # Fallback safety
    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, (N, 2))
        best_radii = np.full(N, 0.05)
        
    # Final strict repair
    best_radii = repair_packing(best_centers, best_radii)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
