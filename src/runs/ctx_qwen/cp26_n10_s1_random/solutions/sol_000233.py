# sol_000233 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000142 (state d65765d5) state=4b6f20f2 sum of radii=2.628008 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def constraints(vars_arr, n):
    """Inequality constraints for SLSQP: must be >= 0."""
    x = vars_arr[:n]
    y = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    c = []
    # Boundary constraints: circle inside [0,1]x[0,1]
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap: squared distance >= squared sum of radii
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dr = r[idx_i] + r[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c)

def objective(vars_arr, n):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_arr[2*n:])

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        # Max radius limited by distance to nearest boundary
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    # Variable bounds: x,y in [0,1], r in [1e-6, 0.5]
    bounds_vars = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Generate diverse hexagonal initial configurations
    starts = []
    row_pats = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], 
        [6,6,4,6,4], [5,5,6,5,5], [6,4,6,4,6],
        [5,6,6,5,4], [6,5,4,6,5]
    ]
    
    for pat in row_pats:
        if sum(pat) != n: 
            continue
        pts = []
        r0 = 0.095
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx%2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: 
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:n])
        starts.append(pts)
        
        # Add perturbed variants to escape symmetry traps
        for _ in range(4):
            p = pts + rng.uniform(-0.015, 0.015, pts.shape)
            p = np.clip(p, 0.05, 0.95)
            starts.append(p)
            
    # Phase 2: Joint SLSQP optimization + LP refinement
    for s in starts:
        x0 = np.zeros(3*n)
        x0[:n] = s[:,0]
        x0[n:2*n] = s[:,1]
        
        # Smart initial radii estimation based on local geometry
        r_init = np.full(n, 0.085)
        for i in range(n):
            dw = min(s[i,0], 1.0-s[i,0], s[i,1], 1.0-s[i,1])
            dn = np.min(np.sqrt(np.sum((s - s[i])**2, axis=1)))
            r_init[i] = min(dw, dn / 2.0) * 0.9
        x0[2*n:] = r_init
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_vars, 
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                c_mat = np.column_stack((cx, cy))
                
                # LP refinement extracts optimal radii for these centers
                r_lp, s_lp = solve_radii_lp(c_mat)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_mat.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # Fallback in case all optimizations fail
    if best_centers is None:
        best_centers = np.tile(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)
        best_centers = np.hstack([best_centers, np.repeat(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)])
        best_centers = np.vstack([best_centers, [[0.5, 0.5]]])
        best_radii, best_sum = solve_radii_lp(best_centers)

    # Phase 3: Stochastic hill-climbing on centers evaluated via LP
    # Alternating center perturbation and exact LP radius extraction escapes local minima
    if best_centers is not None:
        for step in range(600):
            i = rng.integers(n)
            old_c = best_centers[i].copy()
            # Adaptive step size that decays over time
            step_size = 0.014 * (0.96 ** (step / 40.0))
            best_centers[i] += rng.uniform(-step_size, step_size, 2)
            best_centers[i] = np.clip(best_centers[i], 1e-4, 1.0 - 1e-4)
            
            r_try, s_try = solve_radii_lp(best_centers)
            if s_try > best_sum:
                best_sum = s_try
                best_radii = r_try.copy()
            else:
                best_centers[i] = old_c
                
    # Phase 4: Final strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i,0] - best_centers[j,0], 
                         best_centers[i,1] - best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    # Apply scaling with a tiny buffer to strictly satisfy checker tolerance
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
