# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000055 (state 4605f88a) state=16cb787f sum of radii=2.630713 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

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
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N

def generate_init(seed, angle_deg=0.0, r_base=0.095):
    """Generate a rotated hexagonal lattice initialization with safe feasible radii."""
    np.random.seed(seed)
    pts = []
    y = 0.0
    row = 0
    # Generate slightly more than N points to allow rotation/filtering loss
    while len(pts) < N + 15:
        x = (row % 2) * r_base
        while x <= 1.0 + r_base:
            pts.append([x, y])
            x += 2.0 * r_base
        y += np.sqrt(3.0) * r_base
        row += 1
        
    pts = np.array(pts[:N+15])
    
    # Rotate and center around (0.5, 0.5)
    center = np.array([0.5, 0.5])
    pts -= center
    ang = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts @ rot.T + center
    
    # Filter points strictly inside the square
    mask = (pts[:,0]>=0.05) & (pts[:,0]<=0.95) & (pts[:,1]>=0.05) & (pts[:,1]<=0.95)
    pts = pts[mask]
    
    # Pad if rotation/filtering removed too many points
    if len(pts) < N:
        pad = N - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    pts = pts[:N]
    
    # Add small perturbation to break exact symmetry
    pts += np.random.uniform(-0.002, 0.002, pts.shape)
    
    # Compute safe initial radii to guarantee feasibility
    radii = np.zeros(N)
    for i in range(N):
        min_d = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        for j in range(N):
            if i != j:
                d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d * 0.35  # Generous safety margin
        
    x0 = np.zeros(3*N)
    x0[0::3] = pts[:,0]
    x0[1::3] = pts[:,1]
    x0[2::3] = radii
    return x0

def run_packing():
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Extensive multi-start with varied rotations and densities
    configs = []
    for seed in range(15):
        configs.append(generate_init(seed, angle_deg=0.0, r_base=0.095))
    for seed in range(12):
        configs.append(generate_init(seed, angle_deg=15.0, r_base=0.092))
    for seed in range(10):
        configs.append(generate_init(seed, angle_deg=30.0, r_base=0.098))
    for seed in range(8):
        configs.append(generate_init(seed, angle_deg=45.0, r_base=0.090))
        
    for x0 in configs:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 25000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Deflation & Perturbation Refinement to escape local minima
    if best_x is not None:
        for _ in range(8):
            x0 = best_x.copy()
            # Shrink radii slightly to free up space for repositioning
            x0[2::3] *= 0.985
            # Perturb centers to explore new topologies
            x0[0::3] += np.random.normal(0, 0.0015, N)
            x0[1::3] += np.random.normal(0, 0.0015, N)
            
            # Project perturbed variables back to strict bounds
            for i in range(N):
                r = max(0.005, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0-r)
                x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0-r)
                x0[3*i+2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 30000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-7 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization completely fails
    if best_x is None:
        best_x = generate_init(0, 0.0, 0.09)
        best_sum = -objective(best_x)
        
    # Extract centers and radii
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Phase 3: Strict Validation & Minimal Numerical Repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if (radii[i] < 0 or centers[i,0] < radii[i]-1e-10 or centers[i,0] > 1.0-radii[i]+1e-10 or 
                centers[i,1] < radii[i]-1e-10 or centers[i,1] > 1.0-radii[i]+1e-10):
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if d < radii[i]+radii[j]-1e-10:
                        valid = False; break
                if not valid: break
        if valid: break
        
        # Gentle shrinkage to guarantee strict compliance without sacrificing much sum
        radii *= 0.9995
        centers[:,0] = np.clip(centers[:,0], radii, 1.0-radii)
        centers[:,1] = np.clip(centers[:,1], radii, 1.0-radii)
        
    return centers, radii, float(np.sum(radii))
