# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000002 (state 2c120403) state=a90c8719 sum of radii=2.603303 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26

def objective(p):
    """Objective: minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(p[2::3])

def constraint_func(p):
    """
    Computes all boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0.
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0, r >= 0
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r, r])
    
    # Vectorized pairwise overlap constraints: dist(i,j) - (r_i + r_j) >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    # Extract upper triangle (i < j) to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dists = np.hypot(dx, dy)[mask]
    sums_r = dr[mask]
    
    return np.concatenate([c, dists - sums_r])

def get_initial_config(seed):
    """
    Generates a feasible, tightly-packed initial configuration using 
    iterative growth and force-directed relaxation.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a hexagonal lattice pattern
    pts = []
    for row in range(6):
        for col in range(6):
            if len(pts) >= N:
                break
            x = col * 2.0 + (row % 2) * 1.0
            y = row * np.sqrt(3)
            pts.append([x, y])
    pts = np.array(pts[:N])
    
    # Scale and center to fit within the unit square with margin
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0)) * 0.8 + 0.1
    
    centers = pts.copy()
    radii = np.full(N, 0.03)
    
    # Force-directed relaxation with uniform radius growth
    for _ in range(400):
        radii *= 1.0003
        forces = np.zeros_like(centers)
        
        for i in range(N):
            # Inter-circle repulsion
            for j in range(i + 1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = np.hypot(dx, dy)
                min_d = radii[i] + radii[j]
                if d < min_d and d > 1e-9:
                    f = (min_d - d) * 5.0 / d
                    fx, fy = dx * f, dy * f
                    forces[i,0] += fx
                    forces[i,1] += fy
                    forces[j,0] -= fx
                    forces[j,1] -= fy
                    
            # Boundary repulsion
            cx, cy = centers[i]
            r = radii[i]
            if cx < r: forces[i,0] += (r - cx) * 10.0
            if cx > 1-r: forces[i,0] -= (cx - (1-r)) * 10.0
            if cy < r: forces[i,1] += (r - cy) * 10.0
            if cy > 1-r: forces[i,1] -= (cy - (1-r)) * 10.0
            
        centers += forces * 0.0005
        centers = np.clip(centers, 0.01, 0.99)
        
    # Ensure strict feasibility before returning to optimizer
    for _ in range(100):
        min_gap = np.inf
        for i in range(N):
            cx, cy = centers[i]
            r = radii[i]
            min_gap = min(min_gap, cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r)
            for j in range(N):
                if i != j:
                    d = np.hypot(cx - centers[j,0], cy - centers[j,1])
                    min_gap = min(min_gap, d - r - radii[j])
        if min_gap < -1e-7:
            radii -= min(0.002, abs(min_gap)/2)
        else:
            break
    radii = np.maximum(radii, 0.01)
        
    p = np.zeros(3 * N)
    p[0::3] = centers[:, 0]
    p[1::3] = centers[:, 1]
    p[2::3] = radii
    return p

def run_packing() -> tuple:
    best_p = None
    best_val = np.inf
    
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    cons = opt.NonlinearConstraint(constraint_func, 0, np.inf)
    
    # Multiple restarts to escape local minima
    for seed in range(15):
        p0 = get_initial_config(seed)
        try:
            res = opt.minimize(
                objective, 
                p0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            if res.fun < best_val:
                # Verify strict feasibility
                if np.all(constraint_func(res.x) >= -1e-9):
                    best_val = res.fun
                    best_p = res.x.copy()
        except Exception:
            pass
            
    # Fallback if optimization failed unexpectedly
    if best_p is None:
        best_p = get_initial_config(0)
        
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = np.maximum(best_p[2::3], 0.0)
    
    # Final repair pass to guarantee validation passes within numerical tolerance
    for _ in range(30):
        changed = False
        
        # Clamp to boundaries
        for i in range(N):
            cx, cy = centers[i]
            r = radii[i]
            mx = min(cx, 1.0 - cx, cy, 1.0 - cy)
            if r > mx + 1e-10:
                radii[i] = mx
                changed = True
        if not changed:
            break
            
        # Resolve overlaps by shrinking radii proportionally
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    shrink = (radii[i] + radii[j] - d) / 2.0
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
