# sol_000397 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000353 (state 4ca32851) state=557c9bff sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure for speed
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

def compute_lp(centers):
    """Solves LP for maximal radii given fixed centers and computes exact subgradient via duals."""
    centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
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
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
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
            if d > 1e-7:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def obj_grad(x):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = x.reshape(N, 2)
    _, s, g = compute_lp(c)
    return -s, -g.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # Hexagonal lattice patterns
    pats = [[6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [5,5,6,5,5], 
            [4,6,6,6,4], [6,4,6,5,5], [5,5,5,5,6], [5,6,4,5,6]]
    for pat in pats:
        for r0 in np.linspace(0.088, 0.112, 8):
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.array(c[:N]))

    # Rotated hexagonal lattices
    for angle in np.linspace(0.0, np.pi/5, 4):
        for r0 in [0.092, 0.100, 0.108]:
            c = []
            y = r0
            row = 0
            while len(c) < N:
                sh = r0 if row % 2 == 1 else 0.0
                x = r0 + sh
                while x + r0 <= 1.0 + 1e-6 and len(c) < N:
                    pt = np.array([x, y])
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    # Rotate around center (0.5, 0.5)
                    pt_rot = np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), pt - 0.5) + 0.5
                    c.append(pt_rot.tolist())
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
                row += 1
            starts.append(np.clip(np.array(c[:N]), 0.02, 0.98))

    # Force-repelled random starts
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(400):
            diffs = c[:, None, :] - c[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            dists = np.maximum(dists, 1e-6)
            inv_d2 = 1.0 / (dists**2 + 1e-5)
            np.fill_diagonal(inv_d2, 0.0)
            f = np.zeros_like(c)
            for d in range(2):
                f[:, d] = np.sum(diffs[:, :, d] * inv_d2, axis=1) * 0.006
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)

    # Corner-biased starts
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        c += rng.normal(0, 0.01, c.shape)
        starts.append(np.clip(c, 0.02, 0.98))
        
    return starts

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared distances)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def slsqp_polish(centers, radii):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        # Boundary clamping
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        # Pairwise shrinkage
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
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
    
    best_c = None
    best_r = None
    best_s = -1.0
    bounds_c = [(0.001, 0.999)] * (2 * N)
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B with exact LP gradients
    for c_init in starts:
        # Add tiny noise to break symmetry
        c_init += rng.normal(0, 1e-4, c_init.shape)
        c_init = np.clip(c_init, 0.01, 0.99)
        
        try:
            res = minimize(obj_grad, c_init.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 6000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = compute_lp(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_s, _ = compute_lp(best_c)
        
    # Phase 2: Perturbation & Local Search to escape local minima
    for step in range(120):
        scale = 0.022 * (0.91 ** (step // 8))
        c_pert = best_c.copy()
        
        # Randomized perturbation strategy
        strategy = rng.integers(0, 3)
        if strategy == 0:
            idx = rng.choice(N, size=rng.integers(2, 6), replace=False)
            c_pert[idx] += rng.normal(0, scale, (len(idx), 2))
        elif strategy == 1:
            i, j = rng.choice(N, 2, replace=False)
            c_pert[i], c_pert[j] = c_pert[j].copy(), c_pert[i].copy()
        else:
            c_pert += rng.normal(0, scale * 0.5, c_pert.shape)
            
        c_pert = np.clip(c_pert, 0.01, 0.99)
        
        try:
            res = minimize(obj_grad, c_pert.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = compute_lp(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Simulated Annealing for global refinement
    c_sa = best_c.copy()
    s_sa = best_s
    T = 0.010
    for step in range(1000):
        k = rng.integers(1, 4)
        idx = rng.choice(N, size=k, replace=False)
        c_try = c_sa.copy()
        c_try[idx] += rng.normal(0, 0.006 * np.sqrt(T), (k, 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        _, s_try, _ = compute_lp(c_try)
        delta = s_try - s_sa
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_sa, s_sa = c_try, s_try
            if s_sa > best_s:
                best_s = s_sa
                best_c = c_sa.copy()
                best_r, _, _ = compute_lp(best_c)
        T *= 0.995
        
    # Local polish after SA
    try:
        res = minimize(obj_grad, best_c.flatten(), method='L-BFGS-B', jac=True,
                       bounds=bounds_c, options={'maxiter': 4000, 'ftol': 1e-13})
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt, _ = compute_lp(c_opt)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
    except Exception:
        pass
        
    # Phase 4: Joint SLSQP Polish for final precision
    c_final, r_final, s_final = slsqp_polish(best_c, best_r)
    if s_final > best_s:
        best_c = c_final
        best_r = r_final
        best_s = s_final
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    return best_c, radii, final_sum
