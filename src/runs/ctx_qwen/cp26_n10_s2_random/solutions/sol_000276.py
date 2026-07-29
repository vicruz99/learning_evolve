# sol_000276 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000237 (state 963256f0) state=bf51a6a4 sum of radii=2.627681 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute pairwise indices for efficiency
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
N_PAIRS = len(PAIR_I)

def solve_lp_and_gradient(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, sum_radii, and gradient w.r.t centers.
    """
    n = centers.shape[0]
    
    # Boundary upper bounds: r_i <= min(x, 1-x, y, 1-y)
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-7)
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 0.0)
    
    b_ub = dists[PAIR_I, PAIR_J]
    
    # LP constraint matrix structure: r_i + r_j <= dist(i,j)
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), PAIR_I] = 1.0
    A_ub[np.arange(N_PAIRS), PAIR_J] = 1.0
    
    # Solve LP: max sum(r) s.t. A_ub @ r <= b_ub, 0 <= r <= ub
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract dual marginals safely
    duals = np.zeros(N_PAIRS)
    if hasattr(res, 'marginals') and res.marginals is not None:
        if hasattr(res.marginals, 'ineqlin'):
            duals = res.marginals.ineqlin
        elif hasattr(res.marginals, 'ineq'):
            duals = res.marginals.ineq
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    # Compute gradient from active pairwise constraints
    grad = np.zeros_like(centers)
    active_mask = duals > 1e-8
    if np.any(active_mask):
        active_i = PAIR_I[active_mask]
        active_j = PAIR_J[active_mask]
        active_duals = duals[active_mask]
        
        d_vec = centers[active_i] - centers[active_j]
        d_norm = dists[active_i, active_j]
        # Avoid division by zero
        safe_norm = np.maximum(d_norm, 1e-9)
        unit_vec = d_vec / safe_norm[:, np.newaxis]
        
        # grad[i] += lambda * unit_vec, grad[j] -= lambda * unit_vec
        np.add.at(grad, active_i, unit_vec * active_duals[:, np.newaxis])
        np.add.at(grad, active_j, -unit_vec * active_duals[:, np.newaxis])
        
    return radii, np.sum(radii), grad

def generate_inits(rng):
    """Generates diverse starting configurations focusing on boundaries and efficient interior packing."""
    configs = []
    
    # 1. Corner and edge biased starts
    for _ in range(10):
        c = np.zeros((N, 2))
        # Place 4 in corners
        c[0] = [0.12, 0.12]
        c[1] = [0.88, 0.12]
        c[2] = [0.12, 0.88]
        c[3] = [0.88, 0.88]
        # Place 8 along edges (2 per edge)
        c[4:6] = [[0.38, 0.12], [0.62, 0.12]]
        c[6:8] = [[0.88, 0.38], [0.88, 0.62]]
        c[8:10] = [[0.38, 0.88], [0.62, 0.88]]
        c[10:12] = [[0.12, 0.38], [0.12, 0.62]]
        # Fill remaining 14 with hexagonal-ish interior
        pts = []
        y = 0.3
        while len(pts) < 14:
            x = 0.25 if len(pts) % 2 == 0 else 0.32
            while x < 0.75 and len(pts) < 14:
                pts.append([x, y])
                x += 0.14
            y += 0.12
        c[12:] = pts[:14]
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.05, 0.95)
        configs.append(c)

    # 2. Hexagonal lattice patterns with varying densities
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [6, 5, 5, 5, 5]
    ]
    for pat in patterns:
        for scale in [0.95, 1.0, 1.05]:
            c = []
            r0 = 0.095 * scale
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
            
    # 3. Force-directed repulsion starts
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if 0.0 < dist < 0.18:
                        push = (0.18 - dist) * 0.02 / (dist + 1e-6)
                        forces[i] += d_vec / dist * push
                        forces[j] -= d_vec / dist * push
            c += forces
            c = np.clip(c, 0.05, 0.95)
        configs.append(c)
        
    return configs

def objective_center_only(v, rng=None):
    """Objective for center optimization: minimize negative sum of radii."""
    c = v.reshape(N, 2)
    _, s, _ = solve_lp_and_gradient(c)
    return -s

def gradient_center_only(v):
    """Gradient for center optimization."""
    c = v.reshape(N, 2)
    _, _, g = solve_lp_and_gradient(c)
    return -g.flatten()

def optimize_centers_lbfgs(c0, rng, max_iter=2000):
    """Optimizes centers using L-BFGS-B with exact LP gradient."""
    bounds = [(0.001, 0.999)] * (2 * N)
    res = minimize(objective_center_only, c0.flatten(), jac=gradient_center_only,
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
    return res.x.reshape(N, 2), -res.fun

def joint_slsqp_polish(centers, radii):
    """Joint SLSQP refinement for precise constraint satisfaction."""
    v0 = np.concatenate([centers.flatten(), radii])
    
    def obj(v):
        return -np.sum(v[2*N:])
        
    def cons(v):
        cc = v[:2*N].reshape(N, 2)
        rr = v[2*N:]
        con = np.concatenate([
            cc[:, 0] - rr,
            1.0 - cc[:, 0] - rr,
            cc[:, 1] - rr,
            1.0 - cc[:, 1] - rr
        ])
        # Pairwise constraints
        dx = cc[PAIR_I, 0] - cc[PAIR_J, 0]
        dy = cc[PAIR_I, 1] - cc[PAIR_J, 1]
        dr = rr[PAIR_I] + rr[PAIR_J]
        con = np.concatenate([con, dx**2 + dy**2 - dr**2])
        return con
        
    bounds = [(0.001, 0.999)] * (2*N) + [(0.0, 0.49)] * N
    
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-9:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], np.sum(res.x[2*N:])
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def repair_packing(centers, radii):
    """Deterministically shrinks radii to guarantee strict validation compliance."""
    radii = radii.copy()
    
    for _ in range(80):
        changed = False
        # Boundary clamp
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-13:
                radii[i] = mr
                changed = True
                
        # Pairwise overlap resolution
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-13:
                    shrink = (req - d) / 2.0 + 1e-10
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
    
    # Phase 1: Generate diverse starts and optimize centers with L-BFGS-B
    configs = generate_inits(rng)
    candidates = []
    
    for c0 in configs:
        try:
            c_opt, s_opt = optimize_centers_lbfgs(c0, rng)
            radii, _, _ = solve_lp_and_gradient(c_opt)
            candidates.append((c_opt, radii, s_opt))
        except Exception:
            continue
            
    # Sort and keep top candidates
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Phase 2: Joint SLSQP Polish on top candidates
    for c_opt, r_opt, s_opt in candidates[:8]:
        c_pol, r_pol, s_pol = joint_slsqp_polish(c_opt, r_opt)
        if s_pol > best_sum:
            best_sum = s_pol
            best_centers = c_pol
            best_radii = r_pol
            
    # Phase 3: Basin Hopping / Simulated Annealing on best solution
    if best_centers is not None:
        c_curr = best_centers.copy()
        r_curr = best_radii.copy()
        s_curr = best_sum
        T = 0.005
        
        for step in range(500):
            T *= 0.992
            # Perturb centers
            c_new = c_curr + rng.normal(0, T, c_curr.shape)
            c_new = np.clip(c_new, 0.02, 0.98)
            
            # Optimize centers from perturbed state
            try:
                c_opt, s_opt = optimize_centers_lbfgs(c_new, rng, max_iter=500)
                r_opt, _, _ = solve_lp_and_gradient(c_opt)
                
                # Accept if better or probabilistically
                if s_opt > s_curr or rng.random() < np.exp((s_opt - s_curr) / max(T * 5.0, 1e-8)):
                    c_curr = c_opt
                    r_curr = r_opt
                    s_curr = s_opt
                    
                    if s_curr > best_sum:
                        best_sum = s_curr
                        best_centers = c_curr.copy()
                        best_radii = r_curr.copy()
                        
                        # Polish after finding new best
                        c_p, r_p, s_p = joint_slsqp_polish(best_centers, best_radii)
                        if s_p > best_sum:
                            best_sum = s_p
                            best_centers = c_p
                            best_radii = r_p
            except Exception:
                continue

    # Fallback safety
    if best_centers is None:
        best_centers = rng.uniform(0.2, 0.8, (N, 2))
        best_radii = np.full(N, 0.05)
        best_sum = np.sum(best_radii)
        
    # Final strict repair
    best_radii = repair_packing(best_centers, best_radii)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
