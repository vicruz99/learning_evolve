# sol_000277 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000245 (state 54b2cb54) state=b5ff8849 sum of radii=2.268981 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
A_LP = None
LP_PAIRS = None

def setup_lp_matrices():
    """Pre-construct the constant structure of the LP constraint matrix."""
    global A_LP, LP_PAIRS
    num_pairs = N * (N - 1) // 2
    A_LP = np.zeros((num_pairs + 4 * N, N))
    LP_PAIRS = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_LP[k, i] = 1.0
            A_LP[k, j] = 1.0
            LP_PAIRS.append((i, j))
            k += 1
    for i in range(N):
        base = num_pairs + 4 * i
        A_LP[base, i] = 1.0
        A_LP[base + 1, i] = 1.0
        A_LP[base + 2, i] = 1.0
        A_LP[base + 3, i] = 1.0

setup_lp_matrices()

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and gradient."""
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-15)
    
    b_ub = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in LP_PAIRS:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b_ub[idx] = c[i, 0]; idx += 1
        b_ub[idx] = 1.0 - c[i, 0]; idx += 1
        b_ub[idx] = c[i, 1]; idx += 1
        b_ub[idx] = 1.0 - c[i, 1]; idx += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs', 
                  options={'presolve': True})
                  
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    
    # Extract duals safely across scipy versions
    duals = np.zeros_like(b_ub)
    try:
        duals = np.asarray(res.ineqlin.marginals)
    except (AttributeError, ValueError, TypeError):
        try:
            duals = np.asarray(res.marginals)
        except (AttributeError, ValueError, TypeError):
            pass
            
    grad = np.zeros_like(c)
    idx = 0
    for i, j in LP_PAIRS:
        mu = duals[idx]
        if mu > 1e-10:
            d = dists[i, j]
            if d > 1e-10:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = len(LP_PAIRS)
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
        
    return radii, np.sum(radii), grad

def obj_and_grad(x_flat):
    """Objective and gradient wrapper for scipy optimizer."""
    c = x_flat.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def repair_packing(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
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

def generate_starts(rng, n_starts=40):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5]
    ]
    
    for pat in patterns:
        for r0 in [0.088, 0.093, 0.098, 0.103, 0.108]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.array(c[:N]))
            
    # Force-directed random starts
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        f = (0.22 - d) / (d**2 + 1e-6) * 0.02
                        forces[i] += d_vec * f
                        forces[j] -= d_vec * f
            c += forces
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Grid starts
    for gs in [5, 6]:
        gx = np.linspace(0.1, 0.9, gs)
        gy = np.linspace(0.1, 0.9, gs)
        cx, cy = np.meshgrid(gx, gy)
        g = np.column_stack([cx.flatten(), cy.flatten()])
        if len(g) >= N:
            g += rng.normal(0, 0.01, g.shape)
            starts.append(np.clip(g[:N], 0.05, 0.95))
            
    return starts[:n_starts]

def run_packing() -> tuple:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    bounds_xy = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    starts = generate_starts(rng, 45)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c0 in starts:
        try:
            res = minimize(obj_and_grad, c0.flatten(), jac=True, method='L-BFGS-B',
                          bounds=bounds_xy, options={'maxiter': 4000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
    else:
        best_r, _, _ = solve_lp_and_grad(best_c)

    # Phase 2: Simulated Annealing Perturbation Loop
    current_c = best_c.copy()
    T = 0.012
    for step in range(50):
        c_pert = current_c + rng.normal(0, T, current_c.shape)
        c_pert = np.clip(c_pert, 1e-3, 1.0 - 1e-3)
        
        try:
            res = minimize(obj_and_grad, c_pert.flatten(), jac=True, method='L-BFGS-B',
                          bounds=bounds_xy, options={'maxiter': 3000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_and_grad(c_opt)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
                current_c = best_c.copy()
            elif rng.random() < np.exp((s_opt - best_sum) / max(T, 1e-9)):
                current_c = c_opt
                
        except Exception:
            pass
        T *= 0.96
        
    # Phase 3: Final precise LP evaluation and strict repair
    radii, final_sum, _ = solve_lp_and_grad(best_c)
    if final_sum > best_sum:
        best_sum = final_sum
        best_r = radii
        
    radii = repair_packing(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
