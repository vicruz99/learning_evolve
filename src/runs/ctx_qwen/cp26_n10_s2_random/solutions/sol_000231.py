# sol_000231 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000224 (state 70cd4a3b) state=a3489418 sum of radii=2.295000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient via duals."""
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    num_pairs = n * (n - 1) // 2
    num_vars = n
    num_cons = num_pairs + 4 * n
    
    A_ub = np.zeros((num_cons, num_vars))
    b_ub = np.zeros(num_cons)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs')
        if not res.success:
            return None, None, None
        duals = res.ineqlin.marginals
    except Exception:
        return None, None, None
        
    radii = res.x
    grad = np.zeros_like(centers)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            lam = duals[idx]
            if lam > 1e-9:
                d = dists[i, j]
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
            idx += 1
            
    for i in range(n):
        mu_L = duals[num_pairs + 4 * i]
        mu_R = duals[num_pairs + 4 * i + 1]
        mu_B = duals[num_pairs + 4 * i + 2]
        mu_T = duals[num_pairs + 4 * i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, np.sum(radii), grad

def obj_and_grad(x_flat):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    c = x_flat.reshape(N, 2)
    c = np.clip(c, 1e-5, 1.0 - 1e-5)
    radii, s, grad = solve_lp_and_grad(c)
    if radii is None:
        return 0.0, np.zeros_like(x_flat)
    return -s, -grad.flatten()

def optimize_local(c0, bounds_c, max_iter=500):
    """Performs local optimization using L-BFGS-B."""
    res = minimize(obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                   bounds=bounds_c, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
    return res.x.reshape(N, 2), -res.fun

def generate_inits(rng):
    """Generates diverse initial configurations."""
    inits = []
    
    # Hexagonal lattice patterns
    pats = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [6, 5, 5, 5, 5], [5, 4, 6, 6, 5]
    ]
    for pat in pats:
        c_hex = []
        r_est = 0.095
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(c_hex) < N:
                    c_hex.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        while len(c_hex) < N:
            c_hex.append(rng.uniform(0.2, 0.8, 2))
        inits.append(np.array(c_hex[:N]))
        
    # Uniform random spreads
    for _ in range(20):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    return inits

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_c = [(0.01, 0.99)] * (2 * N)
    
    best_c = None
    best_sum = -1.0
    
    inits = generate_inits(rng)
    
    # Phase 1: Local optimization from diverse starts
    for c0 in inits:
        c_opt, s_opt = optimize_local(c0, bounds_c, max_iter=800)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = inits[0]
        _, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    for step in range(100):
        noise = 0.008 * (0.96 ** (step // 5))
        c_pert = best_c + rng.normal(0, noise, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        c_opt, s_opt = optimize_local(c_pert, bounds_c, max_iter=500)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    # Final LP solve to match radii exactly to best centers
    radii, _, _ = solve_lp_and_grad(best_c)
    
    # Strict numerical repair
    centers = best_c.copy()
    if radii is not None:
        radii = repair(centers, radii)
    else:
        radii = np.zeros(N)
        
    return centers, radii, float(np.sum(radii))
