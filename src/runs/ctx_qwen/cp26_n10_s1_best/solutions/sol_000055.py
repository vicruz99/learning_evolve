# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000030 (state 57c93ce5) state=4605f88a sum of radii=2.627847 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

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
    b = np.concatenate([cx - cr, 1.0 - cx - cr, cy - cr, 1.0 - cy - cr])
    
    # Overlap constraints: N*(N-1)/2
    # Vectorized distance and radius sum calculations
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    idx = np.tril_indices(N_CIRCLES, -1)
    o = dx[idx]**2 + dy[idx]**2 - dr[idx]**2
    
    return np.concatenate([b, o])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES

def hex_init(r0, seed):
    """Generate a perturbed hexagonal lattice initialization."""
    np.random.seed(seed)
    x = np.zeros(3 * N_CIRCLES)
    count = 0
    y = r0
    row = 0
    while count < N_CIRCLES and y + r0 <= 1.0:
        x_start = r0 + (r0 if row % 2 == 1 else 0.0)
        col = 0
        while count < N_CIRCLES:
            cx = x_start + col * 2 * r0
            if cx + r0 > 1.0:
                break
            x[3*count] = cx + np.random.uniform(-0.001, 0.001)
            x[3*count+1] = y + np.random.uniform(-0.001, 0.001)
            x[3*count+2] = r0 + np.random.uniform(-0.001, 0.001)
            count += 1
            col += 1
        y += np.sqrt(3.0) * r0
        row += 1
    while count < N_CIRCLES:
        x[3*count] = np.random.uniform(0.15, 0.85)
        x[3*count+1] = np.random.uniform(0.15, 0.85)
        x[3*count+2] = 0.05
        count += 1
    return x

def force_init(seed):
    """Generate a dense initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    cx = np.random.uniform(0.15, 0.85, N_CIRCLES)
    cy = np.random.uniform(0.15, 0.85, N_CIRCLES)
    cr = np.full(N_CIRCLES, 0.09)
    
    for step in range(1500):
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        # Repulsive force between nearby circles
        mask = (d < 0.28) & (d > 1e-6)
        f = np.zeros_like(d2)
        f[mask] = 0.015 / d2[mask]
        
        fx = np.sum(f * dx, axis=1)
        fy = np.sum(f * dy, axis=1)
        
        # Wall repulsion
        w = 0.1
        fx += np.where(cx < cr, w, 0.0) - np.where(cx > 1.0 - cr, w, 0.0)
        fy += np.where(cy < cr, w, 0.0) - np.where(cy > 1.0 - cr, w, 0.0)
        
        # Adaptive step size
        alpha = 0.03 * (1.0 - step / 1500.0)
        cx += fx * alpha
        cy += fy * alpha
        
        # Keep within bounds
        cx = np.clip(cx, cr, 1.0 - cr)
        cy = np.clip(cy, cr, 1.0 - cr)
        
    x = np.zeros(3 * N_CIRCLES)
    x[0::3] = cx
    x[1::3] = cy
    x[2::3] = cr
    return x

def run_packing():
    n = N_CIRCLES
    bnds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Extensive multi-start optimization
    inits = []
    # Varied hexagonal starts to cover different density basins
    for s in range(30):
        r0 = 0.088 + s * 0.0008
        inits.append(hex_init(r0, seed=s))
    # Force-directed starts for robust, symmetric-breaking configurations
    for s in range(15):
        inits.append(force_init(seed=s))
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            c_vals = constraints(res.x)
            # Accept if feasible within tolerance and improves best sum
            if np.min(c_vals) >= -1e-6 and curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement with decaying noise to escape local minima
    if best_x is not None:
        for i in range(6):
            noise_scale = 1e-3 * (0.7 ** i)
            x0 = best_x + np.random.normal(0, noise_scale, 3 * n)
            
            # Project perturbed variables back to strict bounds
            for k in range(n):
                r = np.clip(x0[3*k + 2], 0.01, 0.5)
                x0[3*k] = np.clip(x0[3*k], r, 1.0 - r)
                x0[3*k + 1] = np.clip(x0[3*k + 1], r, 1.0 - r)
                x0[3*k + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                               options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
                curr_sum = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
            except Exception:
                pass
                
    # Extract centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    # Phase 3: Strict validity check and numerical repair
    valid = True
    for i in range(n):
        if (radii[i] < 0 or 
            centers[i, 0] < radii[i] - 1e-10 or centers[i, 0] > 1.0 - radii[i] + 1e-10 or 
            centers[i, 1] < radii[i] - 1e-10 or centers[i, 1] > 1.0 - radii[i] + 1e-10):
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Minimal shrinkage to guarantee strict compliance
        for _ in range(100):
            radii *= 0.995
            centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
            
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-12:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
                
    return centers, radii, float(np.sum(radii))
