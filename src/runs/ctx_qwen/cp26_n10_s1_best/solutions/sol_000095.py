# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000072 (state e356f834) state=868f525b sum of radii=2.576050 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

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
    idx = np.tril_indices(N_CIRCLES, -1)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    c = np.concatenate([c, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
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
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def force_spread(centers, radii, steps=200):
    """Vectorized force-directed relaxation to spread centers apart."""
    pts = centers.copy()
    rad = radii.copy()
    n = pts.shape[0]
    
    for t in range(steps):
        # Vectorized pairwise distances and forces
        dx = pts[:, None, 0] - pts[None, :, 0]
        dy = pts[:, None, 1] - pts[None, :, 1]
        d = np.hypot(dx, dy)
        d = np.maximum(d, 1e-7)
        
        # Repulsive force, stronger at close range
        mask = (d < 0.35)
        w = np.where(mask, 0.01 / (d**2 + 0.001), 0.0)
        
        fx = np.sum(w * dx, axis=1)
        fy = np.sum(w * dy, axis=1)
        
        # Wall repulsion
        fx += np.where(pts[:, 0] < rad + 0.02, 0.05, 0.0) - np.where(pts[:, 0] > 1.0 - rad - 0.02, 0.05, 0.0)
        fy += np.where(pts[:, 1] < rad + 0.02, 0.05, 0.0) - np.where(pts[:, 1] > 1.0 - rad - 0.02, 0.05, 0.0)
        
        lr = 0.025 * (1.0 - t / steps)
        pts[:, 0] += fx * lr
        pts[:, 1] += fy * lr
        pts = np.clip(pts, 0.005, 0.995)
        
    return pts

def make_hex_init(angle, seed):
    """Generate a rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    r0 = 0.092
    y = r0
    row = 0
    while len(pts) < N_CIRCLES + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N_CIRCLES + 10:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N_CIRCLES + 10])
    
    # Rotate around center
    if angle != 0.0:
        c = np.array([0.5, 0.5])
        ca, sa = np.cos(angle), np.sin(angle)
        pts = ((pts - c) @ np.array([[ca, -sa], [sa, ca]])) + c
        
    # Filter strictly inside and pad if necessary
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N_CIRCLES:
        pts = np.vstack([pts, [np.random.uniform(0.15, 0.85), np.random.uniform(0.15, 0.85)]])
    pts = pts[:N_CIRCLES] + np.random.normal(0, 0.003, (N_CIRCLES, 2))
    return np.clip(pts, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(123)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Diverse multi-start with alternating LP & Force Spread
    inits = []
    for s in range(25):
        inits.append(make_hex_init(angle=s * 0.045, seed=s * 100))
    for s in range(10):
        inits.append(np.random.uniform(0.15, 0.85, (N_CIRCLES, 2)))
        
    for init_pts in inits:
        pts = init_pts.copy()
        rads = np.full(N_CIRCLES, 0.08)
        
        # Alternating refinement
        for _ in range(6):
            rads = solve_lp_radii(pts)
            pts = force_spread(pts, rads, steps=150)
            
        curr_sum = np.sum(rads)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = pts.copy()
            best_radii = rads.copy()
            
    # Phase 2: Simulated Annealing / Random Perturbation to escape local minima
    for step in range(800):
        idx = np.random.randint(N_CIRCLES)
        temp_c = best_centers.copy()
        temp_c[idx] = np.random.uniform(0.10, 0.90, 2)
        temp_r = solve_lp_radii(temp_c)
        
        if np.sum(temp_r) > best_sum:
            best_sum = np.sum(temp_r)
            best_centers = temp_c
            best_radii = temp_r
            # Local refinement after successful move
            for _ in range(3):
                best_radii = solve_lp_radii(best_centers)
                best_centers = force_spread(best_centers, best_radii, steps=80)
                
    # Phase 3: SLSQP Fine-tuning
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    x0 = np.zeros(3 * N_CIRCLES)
    x0[0::3] = best_centers[:, 0]
    x0[1::3] = best_centers[:, 1]
    x0[2::3] = best_radii
    
    # Run SLSQP with slight perturbations to avoid flat regions
    for pert_scale in [0.0, 0.001, 0.0005]:
        x_try = x0.copy()
        if pert_scale > 0:
            x_try[0::3] += np.random.normal(0, pert_scale, N_CIRCLES)
            x_try[1::3] += np.random.normal(0, pert_scale, N_CIRCLES)
            
        # Project to feasible bounds
        for i in range(N_CIRCLES):
            r = max(0.01, x_try[3*i+2])
            x_try[3*i] = np.clip(x_try[3*i], r, 1.0-r)
            x_try[3*i+1] = np.clip(x_try[3*i+1], r, 1.0-r)
            x_try[3*i+2] = r
            
        try:
            res = minimize(objective, x_try, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    best_radii = res.x[2::3].copy()
        except Exception:
            pass
            
    # Phase 4: Strict Validation & Minimal Numerical Repair
    for _ in range(100):
        valid = True
        for i in range(N_CIRCLES):
            if (best_radii[i] < 0 or 
                best_centers[i, 0] < best_radii[i] - 1e-10 or best_centers[i, 0] > 1.0 - best_radii[i] + 1e-10 or 
                best_centers[i, 1] < best_radii[i] - 1e-10 or best_centers[i, 1] > 1.0 - best_radii[i] + 1e-10):
                valid = False
                break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
                    if d < best_radii[i] + best_radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            break
            
        # Gentle shrinkage to guarantee strict compliance
        best_radii *= 0.9995
        best_centers[:, 0] = np.clip(best_centers[:, 0], best_radii, 1.0 - best_radii)
        best_centers[:, 1] = np.clip(best_centers[:, 1], best_radii, 1.0 - best_radii)
        
    return best_centers, best_radii, float(np.sum(best_radii))
