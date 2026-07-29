# sol_000269 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000233 (state 6e4dc188) state=a26b31eb sum of radii=2.602835 correctness=1.0
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

def solve_lp(centers):
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
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub,
                  bounds=[(0, u) for u in ub], method='highs')
    
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
        grad[i, 0] += duals[base] - duals[base + 1]
        grad[i, 1] += duals[base + 2] - duals[base + 3]
        
    return radii, s_sum, grad

def obj_and_jac(c_flat):
    """Objective and Jacobian for center optimization: minimize negative sum of radii."""
    c = c_flat.reshape(N, 2)
    _, s, g = solve_lp(c)
    return -s, -g.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns with varying densities
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], [5,5,6,5,5]]
    for pat in pats:
        for r_est in [0.085, 0.095, 0.105, 0.115]:
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
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # 2. Corner/Edge biased configurations
    for _ in range(10):
        c = np.zeros((N, 2))
        c[0] = [0.12, 0.12]
        c[1] = [0.88, 0.12]
        c[2] = [0.12, 0.88]
        c[3] = [0.88, 0.88]
        c[4:] = rng.uniform(0.15, 0.85, (N-4, 2))
        c += rng.normal(0, 0.02, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # 3. Random dense packs
    for _ in range(8):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

def sa_centers(c_init, rng, steps=1500):
    """Simulated annealing on centers to escape local optima."""
    c = c_init.copy()
    _, best_s, _ = solve_lp(c)
    best_c = c.copy()
    
    T = 0.006
    decay = 0.995
    step_size = 0.008
    
    for _ in range(steps):
        i = rng.integers(N)
        c_try = c.copy()
        c_try[i] += rng.normal(0, step_size, 2)
        c_try = np.clip(c_try, 0.02, 0.98)
        
        _, s_try, _ = solve_lp(c_try)
        delta = s_try - best_s
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c = c_try
            if s_try > best_s:
                best_s = s_try
                best_c = c.copy()
        T *= decay
        if T < 1e-7:
            T = 1e-7
    return best_c

def joint_slsqp(c_init, r_init, rng):
    """Joint optimization of centers and radii using SLSQP."""
    v0 = np.concatenate([c_init.flatten(), r_init])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(v):
        return -np.sum(v[2*N:])
        
    def cons(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        con = []
        con.append(c[:, 0] - r)
        con.append(1.0 - c[:, 0] - r)
        con.append(c[:, 1] - r)
        con.append(1.0 - c[:, 1] - r)
        idx = np.triu_indices(N, 1)
        dx = c[idx[0], 0] - c[idx[1], 0]
        dy = c[idx[0], 1] - c[idx[1], 1]
        dr = r[idx[0]] + r[idx[1]]
        con.append(dx**2 + dy**2 - dr**2)
        return np.concatenate(con)
        
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 5000, 'ftol': 1e-13})
        if np.min(cons(res.x)) >= -1e-6:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:]
    except:
        pass
    return c_init, r_init

def repair(c, r):
    """Deterministic repair to guarantee strict validation compliance."""
    r = r.copy()
    for _ in range(60):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                req = r[i] + r[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    r[i] -= shrink
                    r[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            mr = min(c[i,0], 1.0-c[i,0], c[i,1], 1.0-c[i,1])
            if r[i] > mr + 1e-12:
                r[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(r, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_c = [(0.005, 0.995)] * (2 * N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient ascent from diverse starts using LP duals
    for c0 in starts:
        try:
            res = minimize(obj_and_jac, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-12})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except:
            continue
            
    # Phase 2: Simulated Annealing refinement on centers
    best_c = sa_centers(best_c, rng)
    best_r, best_sum, _ = solve_lp(best_c)
    
    # Phase 3: Perturbation loop to escape local optima
    for _ in range(8):
        c_pert = best_c + rng.normal(0, 0.004, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        c_pert = sa_centers(c_pert, rng, steps=600)
        r_p, s_p, _ = solve_lp(c_pert)
        if s_p > best_sum:
            best_sum = s_p
            best_c = c_pert.copy()
            best_r = r_p.copy()
            
    # Phase 4: Joint SLSQP polish for precision
    best_c, best_r = joint_slsqp(best_c, best_r, rng)
    _, s_j, _ = solve_lp(best_c)
    if s_j > best_sum:
        best_sum = s_j
        best_r = solve_lp(best_c)[0]
        
    # Phase 5: Strict numerical repair
    final_r = repair(best_c.copy(), best_r.copy())
    return best_c, final_r, float(np.sum(final_r))
