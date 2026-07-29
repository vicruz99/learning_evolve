# sol_000379 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000303 (state 682ce44f) state=e53bef66 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N

# Precompute LP constraint matrix structure globally
A_LP = np.zeros((NUM_PAIRS + NUM_BOUND, N))
PAIR_I = np.zeros(NUM_PAIRS, dtype=int)
PAIR_J = np.zeros(NUM_PAIRS, dtype=int)
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_I[idx] = i
        PAIR_J[idx] = j
        idx += 1

for i in range(N):
    A_LP[idx + 4*i, i] = 1.0
    A_LP[idx + 4*i + 1, i] = 1.0
    A_LP[idx + 4*i + 2, i] = 1.0
    A_LP[idx + 4*i + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii and computes exact gradient w.r.t centers using duals."""
    n = centers.shape[0]
    
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(NUM_PAIRS + NUM_BOUND)
    b_ub[:NUM_PAIRS] = dists[np.triu_indices(n, 1)]
    
    for i in range(n):
        b_ub[NUM_PAIRS + 4*i] = centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 1] = 1.0 - centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 2] = centers[i, 1]
        b_ub[NUM_PAIRS + 4*i + 3] = 1.0 - centers[i, 1]
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(n), 0.0, np.zeros_like(centers)
            
        radii = res.x
        
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except AttributeError:
            try:
                duals = np.asarray(res.ineqlin.marginals)
            except Exception:
                duals = np.zeros(len(b_ub))
                
        grad = np.zeros_like(centers)
        
        # Apply duals for pairwise constraints
        active_pairs = np.where(duals[:NUM_PAIRS] > 1e-9)[0]
        if len(active_pairs) > 0:
            i_idx = PAIR_I[active_pairs]
            j_idx = PAIR_J[active_pairs]
            d = dists[np.ix_(i_idx, j_idx)].flatten()
            safe_d = np.where(d > 1e-9, d, 1e-9)
            vec = (centers[i_idx] - centers[j_idx]) / safe_d[:, np.newaxis]
            lam = duals[:NUM_PAIRS][active_pairs][:, np.newaxis]
            grad[i_idx] += vec * lam
            grad[j_idx] -= vec * lam
            
        # Apply duals for boundary constraints
        idx_base = NUM_PAIRS + 4 * np.arange(n)
        grad[:, 0] += duals[idx_base] - duals[idx_base + 1]
        grad[:, 1] += duals[idx_base + 2] - duals[idx_base + 3]
            
        return radii, np.sum(radii), grad
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)

def lbfgs_wrapper(x):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [6, 6, 6, 8], [7, 7, 7, 5], [8, 8, 6, 4]
    ]
    for pat in patterns:
        for r_est in [0.090, 0.095, 0.100, 0.105, 0.110]:
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
            c += rng.normal(0, 0.005, c.shape)
            c = np.clip(c, 0.05, 0.95)
            inits.append(c)
            
    # Corner/Edge biased starts
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        edges = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
        c[:8] = corners + edges
        c += rng.normal(0, 0.02, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    # Force-directed random spreads
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.28 and d > 1e-6:
                        push = (0.28 - d) * 0.05
                        forces[i] += d_vec / d * push
                        forces[j] -= d_vec / d * push
            c += forces
            c = np.clip(c, 0.1, 0.9)
        inits.append(c)
        
    return inits

def slsqp_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for SLSQP: boundaries and non-overlap (squared distances)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.01, 0.99)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Multi-start L-BFGS-B Center Optimization
    inits = generate_starts(rng)
    for c0 in inits:
        try:
            res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 5000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Basin Hopping with Swaps & Boundary Pushes
    for step in range(100):
        c_trial = best_c.copy()
        noise_scale = 0.008 * (0.90 ** (step // 6))
        
        # Gaussian perturbation
        c_trial += rng.normal(0, noise_scale, c_trial.shape)
        
        # Random coordinate swaps to break symmetry traps
        for _ in range(rng.integers(1, 4)):
            p = rng.choice(N, 2, replace=False)
            c_trial[p] = c_trial[p[::-1]]
            
        # Boundary push strategy: explicitly move some circles toward edges/corners
        if step % 3 == 0:
            for _ in range(rng.integers(1, 4)):
                idx = rng.integers(N)
                if rng.random() < 0.5:
                    c_trial[idx, 0] = rng.uniform(0.03, 0.97)
                else:
                    c_trial[idx, 1] = rng.uniform(0.03, 0.97)
                    
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        try:
            res = minimize(lbfgs_wrapper, c_trial.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 3500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: SLSQP Joint Polish for High-Precision Refinement
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(8):
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_slsqp,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if np.min(slsqp_cons(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Final LP Verification & Strict Repair
    lp_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_sum:
        best_r = lp_r
        best_sum = final_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
