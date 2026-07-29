# sol_000207 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000169 (state 623e904f) state=6e3204d1 sum of radii=2.302887 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute pairwise indices
PAIR_IDX = [(i, j) for i in range(N) for j in range(i + 1, N)]
I_IDX = np.array([i for i, j in PAIR_IDX])
J_IDX = np.array([j for i, j in PAIR_IDX])

# Precompute LP constraint matrix structure (constant)
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_IDX):
    A_LP[k, i] = 1.0
    A_LP[k, j] = 1.0
for i in range(N):
    A_LP[NUM_PAIRS + 4*i, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 1, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 2, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 3, i] = 1.0

def solve_lp(centers):
    """Solves LP for maximal radii given fixed centers and returns radii, sum, and gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    dists = np.hypot(dx, dy)
    
    b_ub = np.zeros(A_LP.shape[0])
    b_ub[:NUM_PAIRS] = dists
    for i in range(N):
        b_ub[NUM_PAIRS + 4*i] = centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 1] = 1.0 - centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 2] = centers[i, 1]
        b_ub[NUM_PAIRS + 4*i + 3] = 1.0 - centers[i, 1]
        
    bounds = [(0.0, u) for u in ub]
    
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return np.full(N, 1e-9), 0.0, np.zeros_like(centers)
    except Exception:
        return np.full(N, 1e-9), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract dual variables (marginals) robustly across scipy versions
    try:
        if hasattr(res, 'marginals'):
            duals = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin'):
            duals = np.asarray(res.ineqlin.marginals)
        else:
            duals = np.zeros(A_LP.shape[0])
    except Exception:
        duals = np.zeros(A_LP.shape[0])
        
    grad = np.zeros_like(centers)
    
    # Gradient from pairwise constraints
    for k in range(NUM_PAIRS):
        lam = duals[k]
        if lam > 1e-9:
            d = dists[k]
            if d > 1e-9:
                vec = (centers[I_IDX[k]] - centers[J_IDX[k]]) / d
                grad[I_IDX[k]] += lam * vec
                grad[J_IDX[k]] -= lam * vec
                
    # Gradient from boundary constraints
    for i in range(N):
        grad[i, 0] += duals[NUM_PAIRS + 4*i] - duals[NUM_PAIRS + 4*i + 1]
        grad[i, 1] += duals[NUM_PAIRS + 4*i + 2] - duals[NUM_PAIRS + 4*i + 3]
        
    return radii, np.sum(radii), grad

def objective_lp(v):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    c = v.reshape(N, 2)
    _, val, grad = solve_lp(c)
    return -val, -grad.flatten()

def init_configs(rng):
    """Generates diverse initial center configurations."""
    cfgs = []
    
    # 1. Hexagonal lattice patterns with varying densities
    pats = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [5, 5, 6, 5, 5], [6, 6, 5, 5, 4],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6]
    ]
    for p in pats:
        for r0 in [0.088, 0.093, 0.098, 0.104]:
            c = []
            y = r0
            for ri, cnt in enumerate(p):
                shift = r0 if ri % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            cfgs.append(c)
            
    # 2. Force-directed spread from random starts
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.22 and dist > 1e-5:
                        push = (0.22 - dist) * 0.05 / dist
                        forces[i] += d_vec * push
                        forces[j] -= d_vec * push
            c += forces
            c = np.clip(c, 0.05, 0.95)
        cfgs.append(c)
        
    # 3. Pure random
    for _ in range(10):
        cfgs.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    return cfgs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    bounds_c = [(0.01, 0.99)] * (2 * N)
    cfgs = init_configs(rng)
    
    # Phase 1: L-BFGS-B optimization on LP objective from diverse starts
    for c0 in cfgs:
        try:
            res = minimize(objective_lp, c0.flatten(), method='L-BFGS-B',
                           bounds=bounds_c, jac=True,
                           options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-10})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Decaying perturbation search to escape local minima
    for step in range(20):
        noise = 0.018 * (0.85 ** step)
        c_pert = best_c + rng.normal(0, noise, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        try:
            res = minimize(objective_lp, c_pert.flatten(), method='L-BFGS-B',
                           bounds=bounds_c, jac=True,
                           options={'maxiter': 2500, 'ftol': 1e-14, 'gtol': 1e-10})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 3: Joint SLSQP refinement on top candidate
    def slsqp_obj(v):
        return -np.sum(v[2 * N:])
        
    def slsqp_cons(v):
        c = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
        dx = c[I_IDX, 0] - c[J_IDX, 0]
        dy = c[I_IDX, 1] - c[J_IDX, 1]
        dr = r[I_IDX] + r[J_IDX]
        con.append(dx**2 + dy**2 - dr**2)
        return np.concatenate(con)
        
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_sl,
                           constraints={'type': 'ineq', 'fun': slsqp_cons},
                           options={'maxiter': 8000, 'ftol': 1e-14})
            if np.min(slsqp_cons(res.x)) >= -1e-8:
                c_opt = res.x[:2 * N].reshape(N, 2)
                r_opt = res.x[2 * N:]
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_c = c_opt
                    best_r = r_opt
        except Exception:
            pass
            
    # Phase 4: Strict deterministic repair to guarantee validation compliance
    if best_c is not None:
        r_rep = best_r.copy()
        for _ in range(80):
            changed = False
            # Boundary enforcement
            for i in range(N):
                mr = min(best_c[i, 0], 1.0 - best_c[i, 0], 
                         best_c[i, 1], 1.0 - best_c[i, 1])
                if r_rep[i] > mr - 1e-11:
                    r_rep[i] = mr
                    changed = True
            # Overlap resolution
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(best_c[i, 0] - best_c[j, 0], 
                                 best_c[i, 1] - best_c[j, 1])
                    if r_rep[i] + r_rep[j] > d - 1e-11:
                        shr = (r_rep[i] + r_rep[j] - d) * 0.5 + 1e-10
                        r_rep[i] -= shr
                        r_rep[j] -= shr
                        changed = True
            if not changed:
                break
        r_rep = np.maximum(r_rep, 0.0)
        best_r = r_rep
        
    return best_c, best_r, float(np.sum(best_r))
