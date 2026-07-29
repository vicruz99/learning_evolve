# sol_000263 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000233 (state 6e4dc188) state=eb77a80b sum of radii=2.620776 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)
N_PAIRS = len(IDX_I)

# Precompute constant structure of the LP constraint matrix
A_LP_STRUC = np.zeros((N_PAIRS + 4 * N, N))
for k in range(N_PAIRS):
    A_LP_STRUC[k, IDX_I[k]] = 1.0
    A_LP_STRUC[k, IDX_J[k]] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_LP_STRUC[base, i] = 1.0
    A_LP_STRUC[base + 1, i] = 1.0
    A_LP_STRUC[base + 2, i] = 1.0
    A_LP_STRUC[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP to maximize sum of radii for fixed centers and computes exact gradient via duals."""
    c = np.clip(centers, 1e-8, 1.0 - 1e-8)
    
    # Upper bounds on radii from boundary distances
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise Euclidean distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    # Build RHS of LP constraints
    b_ub = np.empty(N_PAIRS + 4 * N)
    b_ub[:N_PAIRS] = dists[IDX_I, IDX_J]
    idx = N_PAIRS
    for i in range(N):
        b_ub[idx] = c[i, 0]; idx += 1
        b_ub[idx] = 1.0 - c[i, 0]; idx += 1
        b_ub[idx] = c[i, 1]; idx += 1
        b_ub[idx] = 1.0 - c[i, 1]; idx += 1
        
    # Solve LP: maximize sum(radii) <=> minimize -sum(radii)
    res = linprog(-np.ones(N), A_ub=A_LP_STRUC, b_ub=b_ub,
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals (handles different scipy versions gracefully)
    duals = np.zeros(N_PAIRS + 4 * N)
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = np.asarray(res.marginals.ineqlin)
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = np.asarray(res.ineqlin.marginals)
        
    # Compute gradient of sum(radii) w.r.t centers using envelope theorem
    grad = np.zeros_like(c)
    for k in range(N_PAIRS):
        mu = duals[k]
        if mu > 1e-9:
            d = dists[IDX_I[k], IDX_J[k]]
            if d > 1e-9:
                vec = (c[IDX_I[k]] - c[IDX_J[k]]) / d
                grad[IDX_I[k]] += mu * vec
                grad[IDX_J[k]] -= mu * vec
                
    for i in range(N):
        base = N_PAIRS + 4 * i
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
    pats = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], 
        [5,5,6,5,5], [5,4,6,6,5], [6,6,5,5,4], [4,5,6,5,6],
        [6,6,6,4,4], [4,4,6,6,6], [7,5,6,5,3], [5,7,5,5,4],
        [3,6,7,6,4], [6,3,6,7,4], [5,6,4,6,5], [6,5,4,6,5]
    ]
    for pat in pats:
        for r_est in [0.085, 0.092, 0.098, 0.105, 0.110]:
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
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c)
        
    return starts

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Computes boundary and non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    d = np.linalg.norm(c[IDX_I] - c[IDX_J], axis=1)
    con.append(d - (r[IDX_I] + r[IDX_J]))
    return np.concatenate(con)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
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
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-10})
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
        for step in range(50):
            noise = 0.012 * (0.88 ** (step // 4))
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            try:
                res = minimize(obj_and_jac, c_pert.flatten(), method='L-BFGS-B', jac=True,
                               bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-13})
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
        
        for _ in range(6):
            try:
                res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                               constraints={'type': 'ineq', 'fun': cons_joint},
                               options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
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
