# sol_000301 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000287 (state 4c08251c) state=2daf8894 sum of radii=2.624513 correctness=1.0
# stdout(first 200): Phase 1: Testing 172 initial configurations...   Init 0/172, current best: -1.000000   Init 20/172, current best: 1.924665   Init 40/172, current best: 1.980854   Init 60/172, current best: 1.980854  
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N

# Precompute LP constraint matrix structure (constant part)
A_LP_CONST = np.zeros((NUM_PAIRS + NUM_BOUND, N))
PAIR_LIST = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP_CONST[idx, i] = 1.0
        A_LP_CONST[idx, j] = 1.0
        PAIR_LIST.append((i, j))
        idx += 1
for i in range(N):
    A_LP_CONST[4 * i, i] = 1.0
    A_LP_CONST[4 * i + 1, i] = 1.0
    A_LP_CONST[4 * i + 2, i] = 1.0
    A_LP_CONST[4 * i + 3, i] = 1.0


def solve_lp_and_get_duals(centers):
    """Solves LP for optimal radii given fixed centers, returns radii, duals, and sum."""
    n = centers.shape[0]
    
    # Upper bounds from boundaries
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-12)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-24)
    
    # Build RHS
    b = np.zeros(NUM_PAIRS + NUM_BOUND)
    for k, (i, j) in enumerate(PAIR_LIST):
        b[k] = dists[i, j]
    
    for i in range(n):
        b[NUM_PAIRS + 4 * i] = centers[i, 0]
        b[NUM_PAIRS + 4 * i + 1] = 1.0 - centers[i, 0]
        b[NUM_PAIRS + 4 * i + 2] = centers[i, 1]
        b[NUM_PAIRS + 4 * i + 3] = 1.0 - centers[i, 1]
    
    c_obj = -np.ones(n)
    bounds_r = [(0.0, u) for u in ub]
    
    try:
        res = linprog(c_obj, A_ub=A_LP_CONST, b_ub=b, bounds=bounds_r, method='highs')
        if res.success:
            radii = res.x
            try:
                duals = np.asarray(res.marginals.ineqlin)
            except AttributeError:
                try:
                    duals = np.asarray(res.ineqlin.marginals)
                except AttributeError:
                    duals = np.zeros_like(b)
            return radii, duals, np.sum(radii)
    except Exception:
        pass
    
    # Fallback
    r_fb = np.minimum(ub, 0.05)
    return r_fb, np.zeros(NUM_PAIRS + NUM_BOUND), np.sum(r_fb)


def compute_gradient(centers, duals):
    """Computes gradient of sum of radii w.r.t. centers using LP duals."""
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-24)
    
    k = 0
    for i, j in PAIR_LIST:
        lam = duals[k]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
    
    # Boundary duals
    for i in range(n):
        base = NUM_PAIRS + 4 * i
        grad[i, 0] += duals[base] - duals[base + 1]
        grad[i, 1] += duals[base + 2] - duals[base + 3]
    
    return grad


def lp_objective_and_gradient(x_flat):
    """Objective and gradient for L-BFGS-B (minimizing negative sum of radii)."""
    centers = x_flat.reshape(N, 2)
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    radii, duals, obj_val = solve_lp_and_get_duals(centers)
    grad = compute_gradient(centers, duals)
    # We minimize -sum(radii), so return obj and -grad
    return -obj_val, -grad.flatten()


def force_directed_init(seed, n_circles=N, num_iters=1500):
    """Force-directed layout with repulsion."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.15, 0.85, (n_circles, 2))
    
    for _ in range(num_iters):
        forces = np.zeros_like(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion
        mask = dists < 0.2
        inv_d2 = np.where(mask, 1.0 / (dists + 0.01) ** 2, 0.0)
        for d in range(2):
            forces[:, d] = np.sum(diff[:, :, d] * inv_d2, axis=1) * 0.008
        
        # Boundary repulsion
        forces += np.where(c < 0.1, 0.05 * (0.1 - c), 0.0)
        forces -= np.where(c > 0.9, 0.05 * (c - 0.9), 0.0)
        
        c += forces
        c = np.clip(c, 0.05, 0.95)
    
    return c


def hex_init(pattern, r_est, rng):
    """Generate hexagonal lattice initialization."""
    c = []
    y = r_est
    for r_idx, cnt in enumerate(pattern):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
    
    while len(c) < N:
        c.append([0.5 + rng.normal(0, 0.05), 0.5 + rng.normal(0, 0.05)])
    
    c = np.array(c[:N])
    c = np.clip(c, 0.03, 0.97)
    return c


def generate_all_inits(rng):
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal patterns
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [6, 6, 4, 6, 4], [4, 6, 5, 6, 5], [5, 6, 4, 6, 5],
        [7, 5, 6, 8], [6, 7, 7, 6], [7, 6, 7, 6],
        [6, 8, 6, 6], [8, 6, 6, 6], [6, 6, 8, 6],
        [5, 7, 5, 9], [7, 5, 7, 7], [8, 7, 5, 6],
    ]
    
    for pat in patterns:
        if sum(pat) < N:
            continue
        for r_est in [0.09, 0.095, 0.10, 0.105, 0.11, 0.115]:
            c = hex_init(pat, r_est, rng)
            inits.append(c)
    
    # 2. Force-directed layouts
    for s in range(16):
        c = force_directed_init(s, num_iters=2000)
        inits.append(c)
    
    # 3. Corner-biased starts
    corners = np.array([[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]])
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = corners + rng.normal(0, 0.02, (4, 2))
        c = np.clip(c, 0.03, 0.97)
        inits.append(c)
    
    # 4. Edge-biased starts (place circles along edges)
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:6, 0] = 0.08 + rng.uniform(0, 0.03, 6)
        c[6:12, 0] = 0.92 + rng.uniform(0, 0.03, 6)
        c[12:18, 1] = 0.08 + rng.uniform(0, 0.03, 6)
        c[18:24, 1] = 0.92 + rng.uniform(0, 0.03, 6)
        c = np.clip(c, 0.03, 0.97)
        inits.append(c)
    
    # 5. Random starts
    for _ in range(12):
        c = rng.uniform(0.1, 0.9, (N, 2))
        inits.append(c)
    
    return inits


def gradient_ascent_optimize(c0, max_iter=3000, init_step=0.008, rng=None):
    """Custom gradient ascent with adaptive step size."""
    c = c0.copy()
    best_c = c.copy()
    best_sum = -1.0
    
    radii, duals, curr_sum = solve_lp_and_get_duals(c)
    best_sum = curr_sum
    grad = compute_gradient(c, duals)
    
    step = init_step
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-12:
            break
        
        c_new = c + step * grad / g_norm
        c_new = np.clip(c_new, 0.01, 0.99)
        
        radii_new, duals_new, s_new = solve_lp_and_get_duals(c_new)
        grad_new = compute_gradient(c_new, duals_new)
        
        if s_new > curr_sum + 1e-14:
            c = c_new
            grad = grad_new
            curr_sum = s_new
            step = min(step * 1.06, 0.03)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = c.copy()
        else:
            step *= 0.8
        
        if step < 1e-10:
            break
        
        # Periodic jitter
        if rng is not None and k > 0 and k % 400 == 300:
            jitter_scale = 0.004 * max(0.1, 1.0 - k / max_iter)
            c_jit = c + rng.normal(0, jitter_scale, c.shape)
            c_jit = np.clip(c_jit, 0.02, 0.98)
            radii_j, duals_j, s_j = solve_lp_and_get_duals(c_jit)
            if s_j > best_sum - 1e-6:
                c = c_jit
                grad = compute_gradient(c, duals_j)
                curr_sum = s_j
    
    return best_c, best_sum


def obj_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])


def cons_joint(v):
    """Constraints for SLSQP: boundaries and non-overlap (squared distance)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    i_idx, j_idx = np.triu_indices(N, 1)
    dx = c[i_idx, 0] - c[j_idx, 0]
    dy = c[i_idx, 1] - c[j_idx, 1]
    dr = r[i_idx] + r[j_idx]
    con = np.concatenate([con, dx ** 2 + dy ** 2 - dr ** 2])
    return con


def repair_packing(centers, radii):
    """Deterministic repair for strict validation compliance."""
    radii = radii.copy()
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) * 0.5 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0],
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(12345)
    rng = np.random.default_rng(12345)
    
    bounds_lbfgs = [(0.005, 0.995)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Generate initial configurations
    inits = generate_all_inits(rng)
    
    # Phase 1: Gradient ascent from diverse starts
    print(f"Phase 1: Testing {len(inits)} initial configurations...")
    for i, c0 in enumerate(inits):
        if i % 20 == 0:
            print(f"  Init {i}/{len(inits)}, current best: {best_sum:.6f}")
        c_opt, s_opt = gradient_ascent_optimize(c0, max_iter=2500, init_step=0.006, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
    
    if best_c is None:
        best_c = inits[0]
        best_r, _, best_sum = solve_lp_and_get_duals(best_c)
    else:
        best_r, _, best_sum = solve_lp_and_get_duals(best_c)
    
    print(f"Phase 1 best: {best_sum:.6f}")
    
    # Phase 2: L-BFGS-B refinement on best
    print("Phase 2: L-BFGS-B refinement...")
    for _ in range(8):
        c_trial = best_c + rng.normal(0, 0.003, best_c.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        try:
            res = minimize(lp_objective_and_gradient, c_trial.flatten(), method='L-BFGS-B',
                           jac=True, bounds=bounds_lbfgs,
                           options={'maxiter': 5000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, _, s_opt = solve_lp_and_get_duals(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
    print(f"Phase 2 best: {best_sum:.6f}")
    
    # Phase 3: Aggressive basin-hopping
    print("Phase 3: Basin-hopping...")
    for step in range(60):
        scale = 0.012 * (0.90 ** (step // 6))
        c_trial = best_c + rng.normal(0, scale, best_c.shape)
        c_trial = np.clip(c_trial, 0.01, 0.99)
        
        # Try gradient ascent from perturbed start
        c_opt, s_opt = gradient_ascent_optimize(c_trial, max_iter=1500, init_step=0.005, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, best_sum = solve_lp_and_get_duals(best_c)
    
    print(f"Phase 3 best: {best_sum:.6f}")
    
    # Phase 4: SLSQP joint polish
    print("Phase 4: SLSQP joint optimization...")
    v0 = np.concatenate([best_c.flatten(), best_r])
    
    for iter_slsqp in range(6):
        # Slight perturbation before each SLSQP call
        v_pert = v0 + rng.normal(0, 0.001, v0.shape)
        v_pert[:2 * N] = np.clip(v_pert[:2 * N], 0.001, 0.999)
        v_pert[2 * N:] = np.maximum(v_pert[2 * N:], 0.001)
        
        try:
            res = minimize(obj_joint, v_pert, method='SLSQP', bounds=bounds_slsqp,
                           constraints={'type': 'ineq', 'fun': cons_joint},
                           options={'maxiter': 10000, 'ftol': 1e-14})
            if np.min(cons_joint(res.x)) >= -1e-7:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
    print(f"Phase 4 best: {best_sum:.6f}")
    
    # Phase 5: Final LP solve and repair
    print("Phase 5: Final repair...")
    lp_r, _, lp_sum = solve_lp_and_get_duals(best_c)
    if lp_sum > best_sum:
        best_r = lp_r
        best_sum = lp_sum
    
    final_radii = repair_packing(best_c.copy(), best_r.copy())
    final_sum = np.sum(final_radii)
    
    print(f"Final sum of radii: {final_sum:.6f}")
    
    return best_c, final_radii, float(final_sum)
