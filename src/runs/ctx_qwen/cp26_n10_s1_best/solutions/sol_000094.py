# sol_000094 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000072 (state e356f834) state=5b98fcb7 sum of radii=2.626591 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Compute all boundary and non-overlap constraints as a vector >= 0.
    Uses direct distance for non-vanishing gradients at boundaries."""
    cx, cy, cr = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: 4 * N
    c = np.concatenate([cx - cr, 1.0 - cx - cr, cy - cr, 1.0 - cy - cr])
    
    # Overlap constraints: N*(N-1)/2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    # Safe sqrt to avoid NaNs if centers coincide
    dist = np.sqrt(np.maximum(dx**2 + dy**2, 1e-16))
    c = np.concatenate([c, dist[TRIL_IDX] - dr[TRIL_IDX]])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N

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
            b_ub[idx] = dist
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(seed, r0=0.09, angle=0.0):
    """Generate a rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c = np.array([0.5, 0.5])
        ca, sa = np.cos(angle), np.sin(angle)
        pts = ((pts - c) @ np.array([[ca, -sa], [sa, ca]])) + c
        
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, 2)])
    pts = pts[:N]
    pts += np.random.normal(0, 0.002, pts.shape)
    return pts

def make_force_init(seed):
    """Generate a dense initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(800):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-6:
                    ff = 0.012 / (d**2 + 0.001)
                    f[i] -= ff * dx
                    f[j] += ff * dx
            for dim in range(2):
                if pts[i, dim] < 0.15: f[i, dim] += 0.06
                elif pts[i, dim] > 0.85: f[i, dim] -= 0.06
        pts += f * 0.025
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse multi-start optimization
    inits = []
    # Varied hexagonal starts to cover different density basins and orientations
    for s in range(15):
        r0 = 0.085 + s * 0.0015
        inits.append(make_hex_init(s, r0=r0, angle=s * 0.06))
    # Force-directed starts for robust, symmetric-breaking configurations
    for s in range(10):
        inits.append(make_force_init(seed=s))
    # Grid-like starts
    for s in range(5):
        np.random.seed(s)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        # Sort and grid-ify slightly
        pts[:, 0] = np.sort(pts[:, 0])
        pts[:, 1] = np.sort(pts[:, 1])
        inits.append(pts)
        
    for pts in inits:
        r = solve_lp_radii(pts)
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r
        
        # Project to strict bounds
        for i in range(N):
            ri = x0[3*i+2]
            x0[3*i] = np.clip(x0[3*i], ri, 1.0-ri)
            x0[3*i+1] = np.clip(x0[3*i+1], ri, 1.0-ri)
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                vals = constraints(res.x)
                if np.min(vals) >= -1e-8 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Deflation-Perturbation Refinement to escape local minima
    if best_x is not None:
        for step in range(40):
            noise_scale = 0.0025 * (0.88 ** step)
            x0 = best_x.copy()
            # Shrink radii to free up space for repositioning
            x0[2::3] *= 0.975
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            for i in range(N):
                ri = max(0.005, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], ri, 1.0-ri)
                x0[3*i+1] = np.clip(x0[3*i+1], ri, 1.0-ri)
                x0[3*i+2] = ri
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    vals = constraints(res.x)
                    if np.min(vals) >= -1e-8 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
                        
                        # Optional: LP snap after successful improvement
                        c_opt = np.column_stack((best_x[0::3], best_x[1::3]))
                        r_lp = solve_lp_radii(c_opt)
                        if np.sum(r_lp) > best_sum:
                            best_x[2::3] = r_lp
                            best_sum = np.sum(r_lp)
            except Exception:
                pass
                
    # Fallback
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[2::3] = 0.05
        best_x[0::3] = np.tile(np.linspace(0.15, 0.85, 6), 5)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.15, 0.85, 5), 6)[:N]
        best_sum = -objective(best_x)
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Phase 3: Strict validation and minimal numerical repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i,0] < radii[i]-1e-10 or centers[i,0] > 1.0-radii[i]+1e-10 or \
               centers[i,1] < radii[i]-1e-10 or centers[i,1] > 1.0-radii[i]+1e-10:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if d < radii[i]+radii[j]-1e-10:
                        valid = False; break
                if not valid: break
        if valid: break
        
        # Gentle shrinkage to guarantee strict compliance
        radii *= 0.9995
        centers[:,0] = np.clip(centers[:,0], radii, 1.0-radii)
        centers[:,1] = np.clip(centers[:,1], radii, 1.0-radii)
        
    return centers, radii, float(np.sum(radii))
