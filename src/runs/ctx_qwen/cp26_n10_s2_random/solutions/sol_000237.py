# sol_000237 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000223 (state d9ff9b60) state=963256f0 sum of radii=2.633035 correctness=1.0
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
    Returns radii, sum_radii, and gradient w.r.t centers.
    """
    n = centers.shape[0]
    
    # Upper bounds from boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-6)
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 0.0)
    
    b_ub = dists[np.triu_indices(n, k=1)]
    
    # Solve LP: max sum(r) s.t. A r <= b, 0 <= r <= ub
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract dual marginals safely across scipy versions
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
    
    return radii, np.sum(radii), grad

def generate_inits(rng):
    """Generates diverse starting configurations."""
    configs = []
    
    # 1. Hexagonal lattice patterns with varying densities
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [6, 5, 5, 5, 5]
    ]
    for pat in patterns:
        c = []
        r0 = 0.10
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
        c += rng.normal(0, 0.003, c.shape)
        c = np.clip(c, 0.05, 0.95)
        configs.append(c)
        
    # 2. Force-directed repulsion starts
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if 0.0 < dist < 0.15:
                        push = (0.15 - dist) * 0.05 / (dist + 1e-6)
                        forces[i] += d_vec / dist * push
                        forces[j] -= d_vec / dist * push
            c += forces
            c = np.clip(c, 0.05, 0.95)
        configs.append(c)
        
    # 3. Corner-biased starts (often optimal for packings)
    for _ in range(6):
        c = rng.uniform(0.1, 0.9, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        c += rng.normal(0, 0.01, c.shape)
        c = np.clip(c, 0.02, 0.98)
        configs.append(c)
        
    return configs

def optimize_gradient_ascent(c0, rng, max_iter=1500):
    """Adaptive gradient ascent with automatic basin hopping."""
    c = c0.copy()
    step = 0.008
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
            
        # Add boundary repulsion to prevent sticking to walls without gradient signal
        bound_force = np.zeros_like(c)
        for i in range(N):
            if c[i, 0] < 0.025: bound_force[i, 0] += 1.0
            if c[i, 0] > 0.975: bound_force[i, 0] -= 1.0
            if c[i, 1] < 0.025: bound_force[i, 1] += 1.0
            if c[i, 1] > 0.975: bound_force[i, 1] -= 1.0
        grad += 15.0 * bound_force
        
        gn = np.linalg.norm(grad)
        if gn < 1e-12:
            break
            
        # Trial step
        c_new = c + step * grad / gn
        c_new = np.clip(c_new, 0.005, 0.995)
        
        radii_new, sum_new, _ = solve_lp_and_gradient(c_new)
        
        # Adaptive step logic
        if sum_new > curr_sum - 1e-12:
            c = c_new
            step = min(step * 1.08, 0.04)
        else:
            step *= 0.55
            
        # Automatic basin hopping if stuck
        if no_improve > 150:
            scale = 0.003 * (1.0 + 0.5 * np.random.rand())
            c += rng.normal(0, scale, c.shape)
            c = np.clip(c, 0.01, 0.99)
            no_improve = 0
            step = 0.006
            
        if step < 1e-8:
            break
            
    return best_c, best_sum

def polish_slsqp(centers, radii, rng):
    """Joint SLSQP refinement for precise constraint satisfaction."""
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
        
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-9:
            return res.x[:2 * n].reshape(n, 2), res.x[2 * n:], np.sum(res.x[2 * n:])
    except Exception:
        pass
    return centers, radii, np.sum(radii)

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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Phase 1: Diverse starts & Gradient Ascent
    configs = generate_inits(rng)
    candidates = []
    
    for c0 in configs:
        c_opt, s_opt = optimize_gradient_ascent(c0, rng)
        candidates.append((c_opt, s_opt))
        
    # Sort by performance and keep top 5 for intensive polishing
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for c_opt, s_opt in candidates[:5]:
        radii, _, _ = solve_lp_and_gradient(c_opt)
        c_pol, r_pol, s_pol = polish_slsqp(c_opt, radii, rng)
        if s_pol > best_sum:
            best_sum = s_pol
            best_centers = c_pol
            best_radii = r_pol
            
    # Phase 2: Simulated Annealing / Basin Hopping on best
    if best_centers is not None:
        c_curr = best_centers.copy()
        r_curr = best_radii.copy()
        s_curr = best_sum
        T = 0.004
        
        for step in range(600):
            T *= 0.995
            c_new = c_curr + rng.normal(0, T, c_curr.shape)
            c_new = np.clip(c_new, 0.02, 0.98)
            
            radii_new, s_new, _ = solve_lp_and_gradient(c_new)
            
            # Accept if better, or with probability based on temperature
            if s_new > s_curr or rng.random() < np.exp((s_new - s_curr) / max(T * 10.0, 1e-8)):
                c_curr = c_new
                r_curr = radii_new
                s_curr = s_new
                if s_curr > best_sum:
                    best_sum = s_curr
                    best_centers = c_curr.copy()
                    best_radii = r_curr.copy()
                    
            # Occasional SLSQP polish during cooling
            if step % 100 == 0 and step > 0:
                c_p, r_p, s_p = polish_slsqp(c_curr, r_curr, rng)
                if s_p > best_sum:
                    best_sum = s_p
                    best_centers = c_p
                    best_radii = r_p
                    c_curr, r_curr, s_curr = c_p, r_p, s_p

    # Fallback safety
    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, (N, 2))
        best_radii = np.full(N, 0.05)
        best_sum = np.sum(best_radii)
        
    # Final strict repair
    best_radii = repair_packing(best_centers, best_radii)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
