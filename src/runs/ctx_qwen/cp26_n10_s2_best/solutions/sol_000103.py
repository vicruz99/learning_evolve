# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state af044a19) state=45929acd sum of radii=2.038134 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints(v, n, pi, pj):
    """Compute inequality constraints: boundaries and non-overlap."""
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    
    c = np.empty(4*n + len(pi))
    c[:n] = cx - r
    c[n:2*n] = 1.0 - cx - r
    c[2*n:3*n] = cy - r
    c[3*n:4*n] = 1.0 - cy - r
    
    dx = cx[pi] - cx[pj]
    dy = cy[pi] - cy[pj]
    c[4*n:] = np.sqrt(dx**2 + dy**2) - (r[pi] + r[pj])
    return c

def make_feasible_radii(centers, n):
    """Compute strictly feasible initial radii for given centers."""
    r = np.full(n, 0.5)
    for i in range(n):
        mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            if d < 2.0*mr:
                mr = 0.5 * d
        r[i] = 0.85 * mr
    return r

def relax_positions(centers, radii, n, steps=100):
    """Force-directed relaxation to resolve overlaps and boundary violations."""
    for _ in range(steps):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = np.hypot(dx, dy)
                min_d = radii[i] + radii[j] + 1e-7
                if d < min_d and d > 0:
                    shift = (min_d - d) * 0.5
                    nx, ny = dx/d, dy/d
                    centers[i,0] += nx * shift
                    centers[i,1] += ny * shift
                    centers[j,0] -= nx * shift
                    centers[j,1] -= ny * shift
                    moved = True
        if not moved:
            break
    centers[:,0] = np.clip(centers[:,0], radii, 1.0-radii)
    centers[:,1] = np.clip(centers[:,1], radii, 1.0-radii)
    return centers

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pi, pj = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)]*(2*n) + [(0.0, 0.5)]*n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n, pi, pj)}
    
    best_val = -np.inf
    best_x = None
    
    def try_optimize(x0):
        nonlocal best_val, best_x
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val:
                c_val = constraints(res.x, n, pi, pj)
                if np.min(c_val) >= -1e-7:
                    best_val = -res.fun
                    best_x = res.x.copy()
        except:
            pass
            
    starts = []
    
    # 1. Hexagonal lattices with varying offsets
    for seed in range(8):
        np.random.seed(seed)
        r0 = 0.09
        pts = []
        y = r0 + np.random.uniform(-0.01, 0.01)
        row = 0
        while len(pts) < n+5:
            xs = r0 if row%2==0 else 2*r0
            x = xs
            while x <= 1.0-r0 and len(pts) < n+5:
                pts.append([x,y])
                x += 2.0*r0
            y += np.sqrt(3)*r0
            row += 1
        pts = np.array(pts[:n]) + np.random.uniform(-0.015, 0.015, (n,2))
        pts = np.clip(pts, 0.02, 0.98)
        starts.append(pts)
        
    # 2. Square grids
    for seed in range(6):
        np.random.seed(100+seed)
        pts = np.array(np.meshgrid(np.linspace(0.12, 0.88, 6), np.linspace(0.12, 0.88, 5))).T.reshape(-1,2)
        pts = pts[:n] + np.random.uniform(-0.02, 0.02, (n,2))
        pts = np.clip(pts, 0.02, 0.98)
        starts.append(pts)
        
    # 3. Random uniform
    for seed in range(10):
        np.random.seed(200+seed)
        starts.append(np.random.uniform(0.1, 0.9, (n,2)))
        
    # 4. Corner-biased
    for seed in range(6):
        np.random.seed(300+seed)
        pts = np.array([[0.15,0.15], [0.85,0.15], [0.15,0.85], [0.85,0.85]])
        rest = np.random.uniform(0.2, 0.8, (n-4, 2))
        pts = np.vstack([pts, rest]) + np.random.uniform(-0.02, 0.02, (n,2))
        pts = np.clip(pts, 0.02, 0.98)
        starts.append(pts)

    for pts in starts:
        r_init = make_feasible_radii(pts, n)
        x0 = np.concatenate([pts.flatten(), r_init])
        try_optimize(x0)
        
    # Escape & Continuation Loop
    if best_x is not None:
        for step in range(25):
            np.random.seed(400+step)
            cx = best_x[:2*n].reshape(n,2) + np.random.uniform(-0.012, 0.012, (n,2))
            cx = np.clip(cx, 0.02, 0.98)
            
            r_new = best_x[2*n:] * (0.97 - step * 0.003)
            r_new = np.maximum(r_new, 0.01)
            
            # Quick relaxation to resolve major overlaps from perturbation
            cx = relax_positions(cx, r_new, n, steps=30)
            
            # Blend radii to keep some progress
            r_init = make_feasible_radii(cx, n)
            r_init = np.maximum(r_init, r_new * 0.6)
            
            x0 = np.concatenate([cx.flatten(), r_init])
            try_optimize(x0)
            
            # Continuation: slightly grow radii and re-optimize positions
            if step % 5 == 0:
                cx_opt = best_x[:2*n].reshape(n,2)
                r_opt = best_x[2*n:] * 1.015
                cx_opt = relax_positions(cx_opt, r_opt, n, steps=50)
                x0 = np.concatenate([cx_opt.flatten(), r_opt])
                try_optimize(x0)

    if best_x is None:
        best_x = np.zeros(3*n)
        best_x[:2*n] = np.random.uniform(0.2, 0.8, 2*n)
        best_x[2*n:] = 0.05
        
    centers = best_x[:2*n].reshape(n,2)
    radii = best_x[2*n:]
    
    # Strict post-processing
    eps = 1e-9
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0]-eps, 1.0-centers[i,0]-eps, centers[i,1]-eps, 1.0-centers[i,1]-eps)
        
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d - eps:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + eps
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed: break
        
    return centers, radii, float(np.sum(radii))
