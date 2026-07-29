# sol_000221 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000123 (state 101aee21) state=956d327c sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def compute_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes gradient via LP duals."""
    n = centers.shape[0]
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    idx = 0
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            pairs.append((i, j))
            idx += 1

    c_obj = -np.ones(n)
    bounds_r = [(0.0, u) for u in ub]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return np.zeros(n), -1.0, np.zeros_like(centers)

    radii = res.x
    duals = res.marginals.ineqlin if hasattr(res, 'marginals') else res.ineqlin.marginals

    grad = np.zeros_like(centers)
    for k, (i, j) in enumerate(pairs):
        lam = duals[k]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec

    return radii, np.sum(radii), grad

def optimize_lp_centers(centers0, max_iter=3000, init_step=0.004):
    """Gradient ascent on centers using LP duals to maximize sum of radii."""
    centers = centers0.copy()
    best_sum = -1.0
    best_centers = centers.copy()
    step = init_step
    
    for k in range(max_iter):
        radii, curr_sum, grad = compute_lp_and_grad(centers)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
        
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
            
        centers += step * grad / g_norm
        centers = np.clip(centers, 0.001, 0.999)
        
        if k % 100 == 0 and k > 0:
            step *= 0.95
        if step < 1e-8:
            break
            
    return best_centers, best_sum

def polish_slsqp(centers, radii):
    """Jointly optimizes centers and radii using SLSQP for local polishing."""
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
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
        d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
        con.append(d - (r[idx[0]] + r[idx[1]]))
        return np.concatenate(con)
        
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 2000, 'ftol': 1e-14})
        if res.success and np.min(cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], np.sum(res.x[2*N:])
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def make_hex_init(pattern, scale, rng):
    """Generates a hexagonal lattice initialization with a given row pattern."""
    centers = []
    r0 = 0.09 * scale
    y = r0
    for r_idx, cnt in enumerate(pattern):
        shift = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(centers) < N:
                centers.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3)
    while len(centers) < N:
        centers.append(rng.uniform(0.1, 0.9, 2))
    c = np.array(centers[:N])
    c += rng.normal(0, 0.002, c.shape)
    c = np.clip(c, 0.02, 0.98)
    return c

def repair(centers, radii):
    """Deterministically resolves overlaps and boundary violations."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i+1, N):
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
    rng = np.random.default_rng(42)
    best_c = None
    best_sum = -1.0
    
    patterns = [
        [5,5,5,5,6], [5,6,5,6,4], [6,5,6,5,4], [4,6,6,6,4], 
        [6,6,5,5,4], [5,5,6,5,5], [5,4,6,6,5], [6,4,6,5,5]
    ]
    
    inits = []
    for pat in patterns:
        for sc in [0.85, 0.95, 1.05, 1.15]:
            inits.append(make_hex_init(pat, sc, rng))
            
    for _ in range(12):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # Phase 1: Multi-start LP gradient ascent + SLSQP polish
    for c0 in inits:
        c_opt, s_opt = optimize_lp_centers(c0, max_iter=2500, init_step=0.005)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
        r_opt, _, _ = compute_lp_and_grad(c_opt)
        c_pol, r_pol, s_pol = polish_slsqp(c_opt, r_opt * 0.99)
        if s_pol > best_sum:
            best_sum = s_pol
            best_c = c_pol.copy()
            
    # Phase 2: Iterative perturbation to escape local minima
    if best_c is not None:
        for step_idx in range(45):
            noise = 0.009 * (0.92 ** (step_idx // 5))
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            # Swap two random circles to break symmetry traps
            idx = rng.choice(N, 2, replace=False)
            c_pert[idx] = c_pert[idx[::-1]]
            
            c_opt2, s_opt2 = optimize_lp_centers(c_pert, max_iter=1500, init_step=0.003)
            if s_opt2 > best_sum:
                best_sum = s_opt2
                best_c = c_opt2.copy()
                
            r_opt2, _, _ = compute_lp_and_grad(c_opt2)
            c_pol2, r_pol2, s_pol2 = polish_slsqp(c_opt2, r_opt2 * 0.99)
            if s_pol2 > best_sum:
                best_sum = s_pol2
                best_c = c_pol2.copy()
                
    # Fallback
    if best_c is None:
        best_c = inits[0]
        
    # Extract final optimal radii for the best layout
    best_r, final_sum, _ = compute_lp_and_grad(best_c)
    
    # Strict numerical repair
    best_r = repair(best_c, best_r)
    return best_c, best_r, float(np.sum(best_r))
