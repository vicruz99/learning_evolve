# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000070 (state 16cb787f) state=eb68c7ff sum of radii=2.629515 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Compute all boundary and non-overlap constraints as a vector >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    # Boundary constraints: 4 * N
    c = np.concatenate([cx - cr, 1.0 - cx - cr, cy - cr, 1.0 - cy - cr])
    
    # Overlap constraints: N*(N-1)/2
    # Vectorized distance and radius sum calculations
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    idx = np.tril_indices(N, -1)
    c = np.concatenate([c, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-6, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def force_spread(centers, steps=200):
    """Force-directed relaxation to spread points evenly and push to boundaries."""
    pts = centers.copy()
    for _ in range(steps):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.3 and d > 1e-6:
                    rep = 0.01 / (d**2 + 1e-4)
                    f[i] -= rep * dx
                    f[j] += rep * dx
            for k in range(2):
                if pts[i, k] < 0.05: f[i, k] += 0.05
                elif pts[i, k] > 0.95: f[i, k] -= 0.05
        pts += f * 0.05
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def make_hex_init(r0, angle_deg=0.0):
    """Generate a rotated hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    # Generate slightly more than N points to allow rotation/filtering loss
    while len(pts) < N + 20:
        x = (row % 2) * r0
        while x <= 1.0 - r0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N + 20])
    
    # Rotate and center around (0.5, 0.5)
    center = np.array([0.5, 0.5])
    pts -= center
    ang = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts @ rot.T + center
    
    # Filter points strictly inside the square
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    # Pad if rotation/filtering removed too many points
    if len(pts) < N:
        pad = N - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    pts = pts[:N]
    
    # Add small perturbation to break exact symmetry
    pts += np.random.uniform(-0.002, 0.002, pts.shape)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Extensive multi-start with varied rotations, densities, and force-spread layouts
    inits = []
    for ang in np.linspace(-45, 45, 19):
        for r0 in [0.09, 0.095, 0.10, 0.105]:
            pts = make_hex_init(r0, ang)
            inits.append(pts)
            
    for _ in range(10):
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        inits.append(force_spread(pts, 150))
        
    for pts in inits:
        # Use LP to get optimal radii for these centers
        r_lp = solve_lp_radii(pts)
        if r_lp is None:
            r_lp = np.full(N, 0.05)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_lp * 0.95  # Slightly shrink to guarantee strict initial feasibility
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Deflation & Perturbation Refinement to escape local minima
    if best_x is not None:
        for round_idx in range(30):
            x0 = best_x.copy()
            # Deflate radii to free up space for topological rearrangement
            scale = 0.985 - 0.003 * round_idx
            if scale < 0.95:
                scale = 0.95
            x0[2::3] *= scale
            
            # Perturb centers with decaying noise
            noise = 0.002 * (0.9 ** round_idx)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Project perturbed variables back to strict bounds
            for i in range(N):
                r = max(0.01, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-6 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
                        
                        # LP refinement on new centers to snap radii to theoretical max
                        centers_lp = np.column_stack((best_x[0::3], best_x[1::3]))
                        r_lp = solve_lp_radii(centers_lp)
                        if r_lp is not None:
                            curr_lp = np.sum(r_lp)
                            if curr_lp > best_sum:
                                best_x[2::3] = r_lp
                                best_sum = curr_lp
            except Exception:
                pass
                
    # Fallback if optimization completely fails (should not be reached)
    if best_x is None:
        pts = make_hex_init(0.09, 0.0)
        r_lp = solve_lp_radii(pts)
        best_x = np.zeros(3 * N)
        best_x[0::3] = pts[:, 0]
        best_x[1::3] = pts[:, 1]
        best_x[2::3] = r_lp if r_lp is not None else np.full(N, 0.05)
        best_sum = -objective(best_x)
        
    # Extract centers and radii
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Phase 3: Strict Validation & Minimal Numerical Repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if (radii[i] < 0 or centers[i, 0] < radii[i] - 1e-10 or centers[i, 0] > 1.0 - radii[i] + 1e-10 or 
                centers[i, 1] < radii[i] - 1e-10 or centers[i, 1] > 1.0 - radii[i] + 1e-10):
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Gentle shrinkage to guarantee strict compliance without sacrificing much sum
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
