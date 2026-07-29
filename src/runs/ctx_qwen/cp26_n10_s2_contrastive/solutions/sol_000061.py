# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000038 (state cf517c54) state=030a9617 sum of radii=2.431828 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_lp_and_grad(n, pairs, num_pairs, A_ub, c_obj, centers):
    """Solves for optimal radii given fixed centers and computes gradient of sum radii w.r.t centers."""
    cx, cy = centers[:, 0], centers[:, 1]
    
    bounds_r = []
    for i in range(n):
        b = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        bounds_r.append((0.0, max(0.0, b)))
        
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dists = np.sqrt(dx**2 + dy**2 + 1e-15)
    
    b_ub = dists[np.triu_indices(n, k=1)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    
    if not res.success:
        return 1e6, np.ones((n, 2))
        
    val = -res.fun
    lam = res.ineqlin.marginals
    if lam is None:
        lam = np.zeros(num_pairs)
        
    grad = np.zeros((n, 2))
    for k, (i, j) in enumerate(pairs):
        l = lam[k]
        if l > 1e-12:
            vec = centers[i] - centers[j]
            d = dists[i, j]
            if d > 1e-12:
                factor = l / d
                grad[i, :] -= factor * vec
                grad[j, :] += factor * vec
                
    return val, grad

def solve_radii_lp(n, pairs, num_pairs, A_ub, c_obj, centers):
    """Computes optimal radii for fixed centers."""
    cx, cy = centers[:, 0], centers[:, 1]
    bounds_r = []
    for i in range(n):
        b = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        bounds_r.append((0.0, max(0.0, b)))
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dists = np.sqrt(dx**2 + dy**2 + 1e-15)
    b_ub = dists[np.triu_indices(n, k=1)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    if res.success:
        return res.x, -res.fun
    return np.zeros(n), 0.0

def obj_wrapper(x, n, pairs, num_pairs, A_ub, c_obj):
    """Wrapper for minimize: returns objective and gradient."""
    centers = x.reshape(n, 2)
    val, grad = compute_lp_and_grad(n, pairs, num_pairs, A_ub, c_obj, centers)
    return -val, -grad.ravel()

def run_packing():
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    
    A_ub = np.zeros((num_pairs, n))
    for k, (i, j) in enumerate(pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    c_obj = -np.ones(n)
    bounds_centers = [(0.0, 1.0)] * (2 * n)
    
    best_sum = 0.0
    best_centers = None
    
    # Hexagonal initialization
    r_hex = 0.095
    s = 2.0 * r_hex
    hex_pts = []
    y = r_hex
    row = 0
    while len(hex_pts) < n:
        shift = s / 2 if row % 2 == 1 else 0
        x = r_hex + shift
        while x <= 1 - r_hex and len(hex_pts) < n:
            hex_pts.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    base_hex = np.array(hex_pts[:n])
    
    inits = [base_hex.ravel()]
    for seed in range(15):
        rng = np.random.RandomState(seed)
        p = base_hex + rng.normal(0, 0.012, base_hex.shape)
        p = np.clip(p, 0.05, 0.95)
        inits.append(p.ravel())
    for seed in range(10):
        rng = np.random.RandomState(seed + 100)
        p = rng.uniform(0.15, 0.85, (n, 2))
        inits.append(p.ravel())
        
    args = (n, pairs, num_pairs, A_ub, c_obj)
        
    for x0 in inits:
        try:
            res = minimize(obj_wrapper, x0, method='L-BFGS-B', bounds=bounds_centers,
                           args=args, jac=True, options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < -best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(n, 2).copy()
                
                # Local refinement around best found
                for _ in range(3):
                    rng = np.random.RandomState()
                    x_p = best_centers.ravel() + rng.normal(0, 0.005, 2*n)
                    x_p = np.clip(x_p, 0.01, 0.99)
                    res2 = minimize(obj_wrapper, x_p, method='L-BFGS-B', bounds=bounds_centers,
                                    args=args, jac=True, options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-14})
                    if res2.fun < -best_sum:
                        best_sum = -res2.fun
                        best_centers = res2.x.reshape(n, 2).copy()
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = base_hex
        
    radii, final_sum = solve_radii_lp(n, pairs, num_pairs, A_ub, c_obj, best_centers)
    
    cx, cy = best_centers[:, 0], best_centers[:, 1]
    for i in range(n):
        mx = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-10)
            
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = np.sum(radii)
    
    return best_centers, radii, float(final_sum)
