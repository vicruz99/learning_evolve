# sol_000197 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000163 (state a7643fac) state=5bf6b82f sum of radii=2.498923 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PAIR_INDICES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(PAIR_INDICES)
TRIU_INDICES = np.triu_indices(N, k=1)

# Precompute constant structure of the LP constraint matrix
A_ub_structure = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_ub_structure[k, i] = 1.0
    A_ub_structure[k, j] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_ub_structure[base + np.arange(4), i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient via duals."""
    c = np.clip(centers, 1e-8, 1.0 - 1e-8)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.empty(N_PAIRS + 4 * N)
    b_ub[:N_PAIRS] = dists[TRIU_INDICES]
    for i in range(N):
        base = N_PAIRS + 4 * i
        b_ub[base] = c[i, 0]
        b_ub[base + 1] = 1.0 - c[i, 0]
        b_ub[base + 2] = c[i, 1]
        b_ub[base + 3] = 1.0 - c[i, 1]
        
    bounds = [(0.0, u) for u in ub]
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_structure, b_ub=b_ub, 
                      bounds=bounds, method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(c)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(b_ub.shape[0])
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
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

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints_joint(v):
    """Computes boundary and non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    c_i = c[TRIU_INDICES[0]]
    c_j = c[TRIU_INDICES[1]]
    r_i = r[TRIU_INDICES[0]]
    r_j = r[TRIU_INDICES[1]]
    dx = c_i[:, 0] - c_j[:, 0]
    dy = c_i[:, 1] - c_j[:, 1]
    con.append(np.sqrt(dx**2 + dy**2) - (r_i + r_j))
    return np.concatenate(con)

def get_bounds_joint():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def generate_starts(n, rng):
    """Generates a wide variety of initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [7, 7, 6, 6], [6, 7, 6, 7], [7, 6, 7, 6]
    ]
    
    # Hexagonal lattice patterns with varying densities
    for pat in patterns:
        for r_est in [0.085, 0.092, 0.100, 0.108]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:n])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Force-directed initialization to spread points evenly
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (n, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            for i in range(n):
                for j in range(i + 1, n):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.2 and d > 1e-6:
                        f = (0.2 - d) / d
                        forces[i] += f * d_vec
                        forces[j] -= f * d_vec
                        
                # Boundary repulsion
                forces[i, 0] += max(0.0, 0.15 - c[i, 0]) * 15.0
                forces[i, 0] -= max(0.0, 0.15 - (1.0 - c[i, 0])) * 15.0
                forces[i, 1] += max(0.0, 0.15 - c[i, 1]) * 15.0
                forces[i, 1] -= max(0.0, 0.15 - (1.0 - c[i, 1])) * 15.0
                
            c += forces * 0.05
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Dense random starts
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
    return starts

def optimize_gradient(c0, steps=1500, init_step=0.01, rng=None):
    """Runs gradient ascent on centers to maximize sum of radii."""
    c = c0.copy()
    best_c = c.copy()
    best_s = -1.0
    step = init_step
    no_imp = 0
    
    for k in range(steps):
        r, s, g = solve_lp_and_grad(c)
        if s > best_s:
            best_s = s
            best_c = c.copy()
            no_imp = 0
        else:
            no_imp += 1
            
        if no_imp > 50:
            step *= 0.7
        elif no_imp > 20:
            step *= 0.9
            
        if step < 1e-10:
            break
            
        gn = np.linalg.norm(g)
        if gn > 1e-9:
            c += step * g / gn
            
        if rng is not None and k % 200 == 0 and k > 0:
            c += rng.normal(0, 0.001, c.shape)
            
        c = np.clip(c, 0.005, 0.995)
        
    return best_c, best_s

def repair(centers, radii):
    """Iteratively shrinks radii to resolve overlaps and clamps to boundaries."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    bounds_j = get_bounds_joint()
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(N, rng)
    
    # Phase 1: Gradient ascent from diverse starts
    for c0 in starts:
        c_opt, s_opt = optimize_gradient(c0, steps=2000, init_step=0.012, rng=rng)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, _, _ = solve_lp_and_grad(best_c)
    else:
        best_c = starts[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Simulated Annealing on centers
    c_curr = best_c.copy()
    s_curr = best_s
    T = 0.005
    for step in range(1000):
        T *= 0.998
        c_new = c_curr + rng.normal(0, T, c_curr.shape)
        c_new = np.clip(c_new, 0.02, 0.98)
        r_new, s_new, _ = solve_lp_and_grad(c_new)
        
        if s_new > s_curr or rng.random() < np.exp((s_new - s_curr) / max(T * 2.0, 1e-7)):
            c_curr = c_new
            s_curr = s_new
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
                best_r = r_new.copy()
                
    # Phase 3: Perturbation & Gradient restarts to escape local minima
    for _ in range(8):
        c_pert = best_c + rng.normal(0, 0.005, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        c_opt, s_opt = optimize_gradient(c_pert, steps=1500, init_step=0.008, rng=rng)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 4: SLSQP Joint Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(3):
        v_pert = v0 + rng.normal(0, 0.001, v0.shape)
        v_pert = np.clip(v_pert, 0.01, 0.99)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.01, 0.4)
        try:
            res = minimize(objective_joint, v_pert, method='SLSQP', bounds=bounds_j,
                          constraints={'type': 'ineq', 'fun': constraints_joint},
                          options={'maxiter': 5000, 'ftol': 1e-13})
            if np.min(constraints_joint(res.x)) >= -1e-9:
                s = np.sum(res.x[2*N:])
                if s > best_s:
                    best_s = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 5: Final strict repair to guarantee validation passes
    best_r = repair(best_c, best_r)
    
    return best_c, best_r, float(np.sum(best_r))
