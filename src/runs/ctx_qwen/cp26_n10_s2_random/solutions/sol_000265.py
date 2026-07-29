# sol_000265 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000233 (state 6e4dc188) state=e9a4e0c4 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_INDICES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(PAIR_INDICES)

# Precompute constant structure of the LP constraint matrix
A_LP = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_LP[k, i] = 1.0
    A_LP[k, j] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP to maximize sum of radii for fixed centers and computes exact gradient via duals."""
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    # Build RHS of LP constraints: pairwise distances + boundary distances
    b_ub = np.zeros(N_PAIRS + 4 * N)
    idx = 0
    for i, j in PAIR_INDICES:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b_ub[idx] = c[i, 0]; idx += 1
        b_ub[idx] = 1.0 - c[i, 0]; idx += 1
        b_ub[idx] = c[i, 1]; idx += 1
        b_ub[idx] = 1.0 - c[i, 1]; idx += 1
        
    # Solve LP: max sum(r) s.t. r_i + r_j <= dist_ij, r_i <= dist_to_boundary
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub,
                  bounds=[(0, None)] * N, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals (handles different scipy versions)
    duals = np.zeros(N_PAIRS + 4 * N)
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = np.asarray(res.marginals.ineqlin)
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = np.asarray(res.ineqlin.marginals)
        
    # Compute gradient of sum(radii) w.r.t centers using dual variables
    grad = np.zeros_like(c)
    idx = 0
    for i, j in PAIR_INDICES:
        mu = duals[idx]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    for i in range(N):
        base = N_PAIRS + 4 * i
        # Boundary duals contribute to gradient: left boundary pushes right, right pushes left, etc.
        grad[i, 0] += duals[base] - duals[base + 1]
        grad[i, 1] += duals[base + 2] - duals[base + 3]
        
    return radii, s_sum, grad

def obj_and_jac(c_flat):
    """Objective and Jacobian for center optimization: minimize negative sum of radii."""
    c = c_flat.reshape(N, 2)
    radii, s_sum, grad = solve_lp_and_grad(c)
    return -s_sum, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns with varying row counts
    pats = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,5,5,6], [4,6,6,6,4], 
        [5,5,6,5,5], [5,4,6,6,5], [6,6,5,5,4], [4,5,6,5,6],
        [5,3,5,6,7], [6,6,6,6,2], [5,5,4,6,6], [6,4,5,5,6]
    ]
    for pat in pats:
        for r_est in [0.088, 0.095, 0.102, 0.108, 0.112]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # 2. Force-directed repulsion spreads (pushes circles to boundaries naturally)
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-5:
                        push = (0.22 - d) * 0.05
                        f[i] += d_vec / d * push
                        f[j] -= d_vec / d * push
            # Add weak attraction to boundaries to encourage larger radii
            for i in range(N):
                f[i, 0] += 0.01 * (0.5 - c[i, 0])
                f[i, 1] += 0.01 * (0.5 - c[i, 1])
            c += f * 0.1
            c = np.clip(c, 0.06, 0.94)
        starts.append(c)
        
    # 3. Corner and edge biased starts
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[4:8] = [[0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        c += rng.normal(0, 0.008, c.shape)
        c = np.clip(c, 0.04, 0.96)
        starts.append(c)
        
    # 4. Random dense packs
    for _ in range(6):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_c = [(0.005, 0.995)] * (2 * N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Local optimization from diverse starts using LP gradients
    for c0 in starts:
        try:
            res = minimize(obj_and_jac, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-9})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Simulated Annealing style perturbation to escape local minima
    if best_c is not None:
        T_init = 0.008
        for step in range(40):
            T = T_init * (0.90 ** step)
            noise = rng.normal(0, T, best_c.shape)
            c_pert = best_c + noise
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            try:
                res = minimize(obj_and_jac, c_pert.flatten(), method='L-BFGS-B', jac=True,
                               bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-12})
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
                
                # Accept if better, or probabilistically if worse
                delta = s_opt - best_sum
                if delta > 0 or rng.random() < np.exp(delta / max(T * 2.0, 1e-7)):
                    best_sum = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
            except Exception:
                continue
                
    # Phase 3: Joint SLSQP Polish to handle slack and tighten constraints simultaneously
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        
        def obj_joint(v):
            return -np.sum(v[2*N:])
            
        def cons_joint(v):
            c = v[:2*N].reshape(N, 2)
            r = v[2*N:]
            con = []
            con.append(c[:, 0] - r)
            con.append(1.0 - c[:, 0] - r)
            con.append(c[:, 1] - r)
            con.append(1.0 - c[:, 1] - r)
            idx = np.triu_indices(N, 1)
            d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
            con.append(d - (r[idx[0]] + r[idx[1]]))
            return np.concatenate(con)
            
        for _ in range(3):
            try:
                res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                               constraints={'type': 'ineq', 'fun': cons_joint},
                               options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                if np.min(cons_joint(res.x)) >= -1e-8:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
                        v0 = res.x.copy()
            except Exception:
                pass
                
    # Final LP solve to match radii exactly to best centers
    if best_c is not None:
        lp_r, s_lp, _ = solve_lp_and_grad(best_c)
        if s_lp > best_sum:
            best_r = lp_r
            best_sum = s_lp
            
    # Strict numerical repair to guarantee validation passes
    final_r = repair(best_c.copy(), best_r.copy())
    return best_c, final_r, float(np.sum(final_r))
