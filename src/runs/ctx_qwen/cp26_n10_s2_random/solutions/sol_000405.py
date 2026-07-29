# sol_000405 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000353 (state 4ca32851) state=eba990a8 sum of radii=2.610436 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_IDX.append((i, j))
        idx += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes gradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    bounds = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    try:
        duals = res.marginals.ineqlin
    except AttributeError:
        duals = np.zeros(len(b))
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def lbfgs_wrapper(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    centers = x_flat.reshape(N, 2)
    _, s, g = solve_lp_and_grad(centers)
    return -s, -g.flatten()

def slsqp_joint_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_joint_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def bcd_optimize_single(centers, radii, idx):
    """Optimize a single circle's position and radius while others are fixed."""
    mask = np.arange(N) != idx
    fixed_centers = centers[mask]
    fixed_radii = radii[mask]
    
    def obj_bcd(x):
        return -x[2]
        
    def cons_bcd(x):
        xi, yi, ri = x
        c = [xi - ri, 1.0 - xi - ri, yi - ri, 1.0 - yi - ri]
        for j in range(len(fixed_centers)):
            dx = xi - fixed_centers[j, 0]
            dy = yi - fixed_centers[j, 1]
            c.append(dx*dx + dy*dy - (ri + fixed_radii[j])**2)
        return np.array(c)
        
    x0 = [centers[idx, 0], centers[idx, 1], radii[idx]]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)]
    
    try:
        res = minimize(obj_bcd, x0, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'ineq', 'fun': cons_bcd},
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success and res.x[2] > radii[idx] - 1e-9:
            centers[idx, 0] = res.x[0]
            centers[idx, 1] = res.x[1]
            radii[idx] = res.x[2]
    except Exception:
        pass
        
def run_bcd_pass(centers, radii, passes=3):
    """Run multiple passes of Block Coordinate Descent."""
    for _ in range(passes):
        order = np.random.permutation(N)
        for idx in order:
            bcd_optimize_single(centers, radii, idx)
        # Sync radii with exact LP to ensure consistency
        r_lp, _, _ = solve_lp_and_grad(centers)
        radii[:] = r_lp
    return centers, radii

def repair(centers, radii):
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

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
            [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5], [5, 5, 5, 5, 6]]
    for pat in pats:
        for r0 in [0.092, 0.097, 0.102, 0.107]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.01, 0.99))
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        starts.append(c)
        
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (N, 2))
        corners = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
        c[:4] = corners
        starts.append(np.clip(c, 0.01, 0.99))
        
    return starts

def run_packing() -> tuple:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    bounds_lbfgs = [(0.005, 0.995)] * (2 * N)
    
    # Phase 1: Multi-start L-BFGS-B
    for c0 in starts:
        try:
            res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 4000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Simulated Annealing with subset perturbations
    curr_c = best_c.copy()
    curr_s = best_s
    T = 0.008
    for step in range(600):
        scale = 0.015 * (0.98 ** (step / 100.0))
        c_try = curr_c.copy()
        idx = rng.choice(N, size=rng.integers(1, 6), replace=False)
        c_try[idx] += rng.normal(0, scale, (len(idx), 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        try:
            res = minimize(lbfgs_wrapper, c_try.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 1500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            
            delta = s_opt - curr_s
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                curr_c = c_opt.copy()
                curr_s = s_opt
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = r_opt.copy()
        except Exception:
            pass
        T *= 0.995
        
    # Phase 3: Block Coordinate Descent Refinement
    best_c, best_r = run_bcd_pass(best_c.copy(), best_r.copy(), passes=4)
    _, s_bcd, _ = solve_lp_and_grad(best_c)
    if s_bcd > best_s:
        best_s = s_bcd
        best_r = best_r.copy()
        
    # Phase 4: Joint SLSQP Polish for micro-adjustments
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_joint_obj, v0, method='SLSQP', bounds=bounds_sl,
                       constraints={'type': 'ineq', 'fun': slsqp_joint_cons},
                       options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_joint_cons(res.x)) >= -1e-7:
            s_sl = np.sum(res.x[2*N:])
            if s_sl > best_s:
                best_s = s_sl
                best_c = res.x[:2*N].reshape(N, 2)
                best_r = res.x[2*N:]
    except Exception:
        pass
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
