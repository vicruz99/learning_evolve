# sol_000267 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000233 (state 6e4dc188) state=e7d1b599 sum of radii=2.608142 correctness=1.0
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
    c = np.clip(centers, 1e-5, 1.0 - 1e-5)
    
    # Upper bounds for radii based on boundaries
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    dists = np.maximum(dists, 2e-9)
    
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
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub,
                  bounds=[(0, None)] * N, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals safely
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
            vec = (c[i] - c[j]) / d
            grad[i] += mu * vec
            grad[j] -= mu * vec
        idx += 1
        
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

def generate_inits(rng):
    """Generates diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattice patterns with varying densities and row structures
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], 
            [5,5,6,5,5], [5,4,6,6,5], [6,6,5,5,4], [4,5,6,5,6],
            [6,6,6,4,4], [5,5,4,6,6], [4,4,6,6,6]]
    for pat in pats:
        for r_est in [0.085, 0.095, 0.105, 0.115, 0.125]:
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
            c = np.clip(c, 0.02, 0.98)
            inits.append(c)
            
    # 2. Force-directed repulsion spreads from random starts
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
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
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    # 3. Corner and edge biased starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c[4:8] = [[0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        c += rng.normal(0, 0.01, c.shape)
        c = np.clip(c, 0.04, 0.96)
        inits.append(c)
        
    # 4. Pure random dense packs
    for _ in range(8):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return inits

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
    bounds_c = [(0.0005, 0.9995)] * (2 * N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Multi-start optimization
    starts = generate_inits(rng)
    for c0 in starts:
        try:
            res = minimize(obj_and_jac, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-12, 'gtol': 1e-9})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            continue
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Simulated Annealing + Local Search to escape local minima
    T = 0.006
    decay = 0.993
    steps = 0
    max_steps = 250
    
    while steps < max_steps:
        # Perturb best_c
        c_try = best_c + rng.normal(0, 0.003 * (1.0 + T), best_c.shape)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        # Occasionally swap centers of two random circles to break symmetry
        if rng.random() < 0.15:
            i, j = rng.choice(N, 2, replace=False)
            c_try[i], c_try[j] = c_try[j], c_try[i]
            
        # Local optimization
        try:
            res = minimize(obj_and_jac, c_try.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 1200, 'ftol': 1e-12})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            
            delta = s_opt - best_sum
            # Accept if better, or probabilistically if worse
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
                
            # Periodic exploration from fresh random config
            if steps % 35 == 0 and steps > 0:
                c_new = rng.uniform(0.15, 0.85, (N, 2))
                c_new += rng.normal(0, 0.02, c_new.shape)
                c_new = np.clip(c_new, 0.05, 0.95)
                try:
                    res = minimize(obj_and_jac, c_new.flatten(), method='L-BFGS-B', jac=True,
                                   bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-12})
                    c_opt2 = res.x.reshape(N, 2)
                    r_opt2, s_opt2, _ = solve_lp_and_grad(c_opt2)
                    if s_opt2 > best_sum:
                        best_sum = s_opt2
                        best_c = c_opt2.copy()
                        best_r = r_opt2.copy()
                except Exception:
                    pass
                    
            T *= decay
            steps += 1
        except Exception:
            steps += 1
            continue
            
    # Phase 3: Joint SLSQP Polish to handle slack and tighten constraints
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
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if np.min(cons_joint(res.x)) >= -1e-7:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Final LP solve to match radii exactly to best centers
    lp_r, s_lp, _ = solve_lp_and_grad(best_c)
    if s_lp > best_sum:
        best_r = lp_r
        best_sum = s_lp
        
    # Final repair to guarantee strict validation compliance
    final_r = repair(best_c.copy(), best_r.copy())
    return best_c, final_r, float(np.sum(final_r))
