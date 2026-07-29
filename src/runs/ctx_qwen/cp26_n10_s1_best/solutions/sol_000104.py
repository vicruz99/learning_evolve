# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000099 (state 1b9343e7) state=2e74a509 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog, basinhopping

N = 26

def get_bounds_full():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def constraints_full(x):
    """Returns all inequality constraints g(x) >= 0."""
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
    idx = np.tril_indices(N, -1)
    c = np.concatenate([c, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    return c

def objective_full(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
    # Boundary limits for each circle
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 0.0)
    
    # Pairwise overlap constraints: r_i + r_j <= dist_ij
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    dists = np.hypot(dx, dy)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    bounds_r = list(zip(np.zeros(n), lims))
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def bh_objective(x_flat):
    """Objective for Basinhopping: negative sum of optimal radii for given centers."""
    centers = x_flat.reshape(N, 2)
    # Ensure centers stay strictly inside for LP stability
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    r = solve_lp_radii(centers)
    if r is not None:
        return -np.sum(r)
    return 1e6  # Penalty for infeasibility

def generate_hex_init(r0):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 5:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    pts = np.array(pts[:N + 5])
    
    # Filter valid points and pad if necessary
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def generate_force_init(seed):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    pts = np.random.uniform(0.2, 0.8, (N, 2))
    for _ in range(400):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.linalg.norm(dx)
                if d < 0.25 and d > 1e-6:
                    f = 0.012 / (d**2 + 0.001)
                    forces[i] -= f * dx
                    forces[j] += f * dx
            for dim in range(2):
                if pts[i, dim] < 0.15: forces[i, dim] += 0.025
                elif pts[i, dim] > 0.85: forces[i, dim] -= 0.025
        pts += forces * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Basinhopping on centers to find global topology
    inits = [generate_hex_init(0.095), generate_force_init(1), generate_force_init(2)]
    
    for c0 in inits:
        try:
            minimizer_kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 60}}
            ret = basinhopping(bh_objective, c0.flatten(), minimizer_kwargs=minimizer_kwargs,
                               niter=120, stepsize=0.012, T=0.4, 
                               accept_test=lambda kwargs: True,
                               interval=40, niter_success=60)
            
            c_opt = ret.x.reshape(N, 2)
            r_opt = solve_lp_radii(c_opt)
            if r_opt is not None:
                s = np.sum(r_opt)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Gradient-based polishing (SLSQP)
    if best_centers is not None:
        x0 = np.zeros(3 * N)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        # Shrink radii slightly to ensure strict feasibility start
        x0[2::3] = np.maximum(best_radii * 0.995, 0.001)
        
        # Project to bounds
        for i in range(N):
            r = x0[3 * i + 2]
            x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
            
        cons = {'type': 'ineq', 'fun': constraints_full}
        bnds = get_bounds_full()
        
        try:
            res = minimize(objective_full, x0, method='SLSQP', bounds=bnds,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            
            if not np.isnan(res.fun):
                c_vals = constraints_full(res.x)
                if np.min(c_vals) >= -1e-7 and -res.fun > best_sum:
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    best_radii = res.x[2::3].copy()
                    best_sum = -res.fun
        except Exception:
            pass

    # Fallback
    if best_centers is None:
        best_centers = generate_hex_init(0.09)
        best_radii = np.full(N, 0.08)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N):
            if best_radii[i] < 0: valid = False; break
            if best_centers[i, 0] - best_radii[i] < -1e-10 or best_centers[i, 0] + best_radii[i] > 1.0 + 1e-10: valid = False; break
            if best_centers[i, 1] - best_radii[i] < -1e-10 or best_centers[i, 1] + best_radii[i] > 1.0 + 1e-10: valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
                    if d < best_radii[i] + best_radii[j] - 1e-10:
                        valid = False; break
                if not valid: break
        if valid: break
        
        # Minimal shrinkage to recover strict feasibility
        best_radii *= 0.9998
        best_centers[:, 0] = np.clip(best_centers[:, 0], best_radii, 1.0 - best_radii)
        best_centers[:, 1] = np.clip(best_centers[:, 1], best_radii, 1.0 - best_radii)

    return best_centers, best_radii, float(np.sum(best_radii))
