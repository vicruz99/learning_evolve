# sol_000279 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000245 (state 54b2cb54) state=bb5be505 sum of radii=2.606687 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_IND = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP structure globally
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
LP_PAIRS = []
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        LP_PAIRS.append((i, j))
        k += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

SLSQP_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

def solve_lp_radii(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and dual multipliers."""
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in LP_PAIRS:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except (AttributeError, ValueError, TypeError):
            try:
                duals = np.asarray(res.marginals.ineqlin)
            except (AttributeError, ValueError, TypeError):
                duals = np.zeros_like(b)
        return res.x, np.sum(res.x), duals
    return np.zeros(N), 0.0, np.zeros_like(b)

def obj_grad_centers(x_flat):
    """Objective and exact gradient for center optimization using LP duals."""
    centers = x_flat.reshape(N, 2)
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    radii, s, duals = solve_lp_radii(centers)
    if s < 1e-9:
        return 0.0, np.zeros_like(x_flat)
        
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    idx = 0
    for i, j in LP_PAIRS:
        mu = duals[idx]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
        
    return -s, -grad.flatten()

def slsqp_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    i, j = TRIU_IND
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def generate_starts(rng, n_inits=40):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5], [5, 7, 5, 5, 4],
        [5, 5, 5, 5, 5, 1], [6, 5, 5, 5, 5], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [6, 6, 4, 5, 5], [5, 5, 4, 6, 6]
    ]
    
    for pat in patterns:
        for r0 in [0.085, 0.092, 0.098, 0.105, 0.112]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            inits.append(np.array(c[:N]))
            
    # Grid + center pattern
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    cx, cy = np.meshgrid(gx, gy)
    c_grid = np.column_stack((cx.flatten(), cy.flatten()))
    c_plus1 = np.vstack([c_grid, [0.5, 0.5]])
    c_plus1 += rng.normal(0, 0.01, c_plus1.shape)
    inits.append(np.clip(c_plus1, 0.05, 0.95))
    
    # Force-directed layouts
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.25 and d > 1e-6:
                        f = (0.25 - d) / (d**2 + 1e-6) * 0.005
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    # Corner-biased
    for _ in range(6):
        c = rng.uniform(0.25, 0.75, (N, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c += rng.normal(0, 0.005, c.shape)
        inits.append(np.clip(c, 0.05, 0.95))
        
    return inits[:n_inits]

def run_packing() -> tuple:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    inits = generate_starts(rng, 40)
    
    # Phase 1: Multi-start L-BFGS-B on centers using exact LP gradient
    for c0 in inits:
        try:
            res = minimize(obj_grad_centers, c0.flatten(), jac=True, method='L-BFGS-B', 
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            c_opt = res.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
        except Exception:
            continue
            
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum, _ = solve_lp_radii(best_c)

    # Phase 2: Joint SLSQP polish to tightly resolve constraints
    r_lp, _, _ = solve_lp_radii(best_c)
    v0 = np.concatenate([best_c.flatten(), r_lp])
    try:
        res_sl = minimize(slsqp_obj, v0, method='SLSQP', bounds=SLSQP_BOUNDS,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 6000, 'ftol': 1e-14})
        if np.min(slsqp_cons(res_sl.x)) >= -1e-7:
            s_sl = np.sum(res_sl.x[2*N:])
            if s_sl > best_sum:
                best_sum = s_sl
                best_c = res_sl.x[:2*N].reshape(N, 2)
                best_r = res_sl.x[2*N:]
    except Exception:
        pass
        
    # Phase 3: Basin-hopping style perturbation & refinement
    for step_idx in range(20):
        noise_scale = 0.012 * (0.90 ** step_idx)
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, 1e-4, 1.0 - 1e-4)
        
        try:
            res_pb = minimize(obj_grad_centers, c_pert.flatten(), jac=True, method='L-BFGS-B',
                              bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-14})
            c_opt = res_pb.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r, _, _ = solve_lp_radii(best_c)
        except Exception:
            continue
            
        # Occasionally try SLSQP from perturbed state
        if step_idx % 3 == 0:
            r_init = np.full(N, best_sum / N * 0.8)
            v_pert = np.concatenate([c_pert.flatten(), r_init])
            try:
                res_s = minimize(slsqp_obj, v_pert, method='SLSQP', bounds=SLSQP_BOUNDS,
                                 constraints={'type': 'ineq', 'fun': slsqp_cons},
                                 options={'maxiter': 4000, 'ftol': 1e-13})
                if np.min(slsqp_cons(res_s.x)) >= -1e-7:
                    s_s = np.sum(res_s.x[2*N:])
                    if s_s > best_sum:
                        best_sum = s_s
                        best_c = res_s.x[:2*N].reshape(N, 2)
                        best_r = res_s.x[2*N:]
            except Exception:
                pass
                
    # Phase 4: Final LP extraction and strict repair
    r_final, s_final, _ = solve_lp_radii(best_c)
    if s_final > best_sum:
        best_sum = s_final
        best_r = r_final
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
