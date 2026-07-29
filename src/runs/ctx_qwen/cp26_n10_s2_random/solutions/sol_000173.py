# sol_000173 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000147 (state da2cd853) state=96145508 sum of radii=2.604214 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute LP constraint matrix structure (constant for fixed N)
# Pairwise constraints: r_i + r_j <= dist(i, j)
A_ub_pairs = np.zeros((N*(N-1)//2, N))
pair_idx = np.zeros((N*(N-1)//2, 2), dtype=int)
idx = 0
for i in range(N):
    for j in range(i+1, N):
        A_ub_pairs[idx, i] = 1.0
        A_ub_pairs[idx, j] = 1.0
        pair_idx[idx] = [i, j]
        idx += 1

# Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
A_ub_bounds = np.zeros((4*N, N))
for i in range(N):
    A_ub_bounds[4*i, i] = 1.0
    A_ub_bounds[4*i+1, i] = 1.0
    A_ub_bounds[4*i+2, i] = 1.0
    A_ub_bounds[4*i+3, i] = 1.0

A_ub = np.vstack([A_ub_pairs, A_ub_bounds])
NUM_PAIRS = len(pair_idx)
c_obj_lp = -np.ones(N)

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and returns gradient of sum radii w.r.t centers."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(A_ub.shape[0])
    b_ub[:NUM_PAIRS] = dists[pair_idx[:, 0], pair_idx[:, 1]]
    for i in range(N):
        b_ub[NUM_PAIRS + 4*i] = centers[i, 0]
        b_ub[NUM_PAIRS + 4*i+1] = 1.0 - centers[i, 0]
        b_ub[NUM_PAIRS + 4*i+2] = centers[i, 1]
        b_ub[NUM_PAIRS + 4*i+3] = 1.0 - centers[i, 1]
        
    bounds_r = [(0.0, u) for u in ub]
    res = linprog(c_obj_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    
    if not res.success:
        return np.full(N, 0.01), 0.0, np.zeros_like(centers)
        
    radii = res.x
    duals = res.ineqlin.marginals if hasattr(res.ineqlin, 'marginals') else np.zeros(A_ub.shape[0])
    sum_r = -res.fun
    
    # Compute gradient of sum_r w.r.t centers using dual variables
    grad = np.zeros_like(centers)
    
    # Pairwise forces: grad += lambda * (c_i - c_j) / ||c_i - c_j||
    for k in range(NUM_PAIRS):
        lam = duals[k]
        if lam > 1e-7:
            i, j = pair_idx[k]
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            if d > 1e-9:
                fx = lam * dx / d
                fy = lam * dy / d
                grad[i, 0] += fx
                grad[i, 1] += fy
                grad[j, 0] -= fx
                grad[j, 1] -= fy
                
    # Boundary forces
    for i in range(N):
        b_idx = NUM_PAIRS + 4*i
        grad[i, 0] += duals[b_idx] - duals[b_idx+1]
        grad[i, 1] += duals[b_idx+2] - duals[b_idx+3]
        
    return radii, sum_r, grad

def obj_func(v):
    """Objective: minimize negative sum of radii."""
    return -solve_lp_and_grad(v.reshape(N, 2))[1]

def grad_func(v):
    """Gradient of objective w.r.t flattened centers."""
    _, _, g = solve_lp_and_grad(v.reshape(N, 2))
    return g.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations: hexagonal lattices and random."""
    starts = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6],
        [6, 5, 5, 5, 5], [4, 6, 6, 6, 4], [5, 5, 6, 5, 5],
        [6, 4, 6, 5, 5], [5, 7, 5, 5, 4], [6, 6, 5, 4, 5],
        [5, 5, 4, 6, 6], [6, 6, 4, 5, 5], [4, 5, 6, 5, 6]
    ]
    
    for pat in patterns:
        for r0 in [0.085, 0.090, 0.095, 0.100, 0.105]:
            c = []
            y = r0
            for row_idx, count in enumerate(pat):
                shift = r0 if row_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(count):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.02, 0.98)
            starts.append(c)
            
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_c = [(0.01, 0.99)] * (2 * N)
    
    best_c = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: L-BFGS-B optimization on centers with exact LP gradients
    for c0 in starts:
        try:
            res = minimize(obj_func, c0.flatten(), method='L-BFGS-B', jac=grad_func,
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-14})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        _, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Perturbation search to escape local minima
    for _ in range(80):
        c_trial = best_c.copy()
        c_trial += rng.normal(0, 0.0025, c_trial.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        try:
            res = minimize(obj_func, c_trial.flatten(), method='L-BFGS-B', jac=grad_func,
                           bounds=bounds_c, options={'maxiter': 1200, 'ftol': 1e-14})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    # Phase 3: Powell derivative-free polish for robustness
    try:
        res = minimize(obj_func, best_c.flatten(), method='Powell', bounds=bounds_c,
                       options={'maxiter': 5000, 'ftol': 1e-15, 'xtol': 1e-15})
        curr_sum = -res.fun
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_c = res.x.reshape(N, 2).copy()
    except Exception:
        pass
        
    # Get exact radii for best centers
    best_r, _, _ = solve_lp_and_grad(best_c)
    
    # Phase 4: Joint SLSQP polish on (centers, radii) to handle tight constraints
    v0 = np.concatenate([best_c.flatten(), best_r])
    
    def obj_joint(v):
        return -np.sum(v[2*N:])
        
    def cons_joint(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
        idx_i, idx_j = np.triu_indices(N, 1)
        d = np.linalg.norm(c[idx_i] - c[idx_j], axis=1)
        con.append(d - (r[idx_i] + r[idx_j]))
        return np.concatenate(con)
        
    bounds_joint = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    try:
        res_j = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                         constraints={'type': 'ineq', 'fun': cons_joint},
                         options={'maxiter': 8000, 'ftol': 1e-14})
        if -res_j.fun > best_sum - 1e-7:
            best_c = res_j.x[:2*N].reshape(N, 2)
            best_r = res_j.x[2*N:]
            best_sum = -res_j.fun
    except Exception:
        pass
        
    # Phase 5: Strict Deterministic Repair for Validator Compliance
    centers = best_c.copy()
    radii = best_r.copy()
    
    for _ in range(50):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Resolve boundary violations
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    return centers, radii, final_sum
