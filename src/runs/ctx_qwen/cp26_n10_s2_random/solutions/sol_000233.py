# sol_000233 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000224 (state 70cd4a3b) state=6e4dc188 sum of radii=2.630831 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_INDICES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(PAIR_INDICES)

# Precompute constant structure of the LP constraint matrix
A_LP_STRUC = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_LP_STRUC[k, i] = 1.0
    A_LP_STRUC[k, j] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_LP_STRUC[base, i] = 1.0
    A_LP_STRUC[base + 1, i] = 1.0
    A_LP_STRUC[base + 2, i] = 1.0
    A_LP_STRUC[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP to maximize sum of radii for fixed centers and computes exact gradient via duals."""
    c = np.clip(centers, 1e-8, 1.0 - 1e-8)
    
    # Distance to boundaries
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Build RHS of LP constraints
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
        
    # Solve LP
    res = linprog(-np.ones(N), A_ub=A_LP_STRUC, b_ub=b_ub,
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
        
    # Compute gradient of sum(radii) w.r.t centers
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
        mu_L = duals[base]
        mu_R = duals[base + 1]
        mu_B = duals[base + 2]
        mu_T = duals[base + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, s_sum, grad

def obj_and_jac(c_flat):
    """Objective and Jacobian for center optimization: minimize negative sum of radii."""
    c = c_flat.reshape(N, 2)
    radii, s_sum, grad = solve_lp_and_grad(c)
    return -s_sum, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns with varying densities
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], 
            [5,5,6,5,5], [5,4,6,6,5], [6,6,5,5,4], [4,5,6,5,6]]
    for pat in pats:
        for r_est in [0.088, 0.095, 0.102, 0.108]:
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
            
    # 2. Force-directed repulsion spreads
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.24 and d > 1e-5:
                        push = (0.24 - d) * 0.04
                        f[i] += d_vec / d * push
                        f[j] -= d_vec / d * push
            c += f
            c = np.clip(c, 0.08, 0.92)
        starts.append(c)
        
    # 3. Random dense packs
    for _ in range(8):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(80):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
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
            
    # Phase 2: Iterative perturbation & refinement to escape local minima
    if best_c is not None:
        for step in range(35):
            noise = 0.008 * (0.88 ** (step // 6))
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            try:
                res = minimize(obj_and_jac, c_pert.flatten(), method='L-BFGS-B', jac=True,
                               bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-12})
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
            except Exception:
                continue
                
    # Phase 3: Joint SLSQP Polish to handle slack and tighten constraints
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
            
        for _ in range(4):
            try:
                res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                               constraints={'type': 'ineq', 'fun': cons_joint},
                               options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
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
            
    final_r = repair(best_c.copy(), best_r.copy())
    return best_c, final_r, float(np.sum(final_r))
