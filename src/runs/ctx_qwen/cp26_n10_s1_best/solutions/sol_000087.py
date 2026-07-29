# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000070 (state 16cb787f) state=8eeb605d sum of radii=2.623812 correctness=1.0
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
    cx, cy, cr = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: 4 * N
    c = np.concatenate([cx - cr, 1.0 - cx - cr, cy - cr, 1.0 - cy - cr])
    
    # Overlap constraints: N*(N-1)/2
    # Vectorized squared distance and radius sum calculations
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    idx = np.tril_indices(N, -1)
    c = np.concatenate([c, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N

def generate_hex_init(n, angle_deg, r_base=0.10):
    """Generate a rotated hexagonal lattice initialization with safe feasible radii."""
    np.random.seed(42)
    pts = []
    y = 0.0
    row = 0
    # Generate slightly more than N points to allow rotation/filtering loss
    while len(pts) < n + 20:
        x = (row % 2) * r_base
        while x <= 1.0 + r_base:
            pts.append([x, y])
            x += 2.0 * r_base
        y += np.sqrt(3.0) * r_base
        row += 1
        
    pts = np.array(pts[:n+20])
    
    # Rotate and center around (0.5, 0.5)
    center = np.array([0.5, 0.5])
    pts -= center
    ang = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts @ rot.T + center
    
    # Filter points strictly inside the square
    mask = (pts[:,0]>=0.02) & (pts[:,0]<=0.98) & (pts[:,1]>=0.02) & (pts[:,1]<=0.98)
    pts = pts[mask]
    
    # Pad if rotation/filtering removed too many points
    if len(pts) < n:
        pad = n - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    pts = pts[:n]
    
    # Add small perturbation to break exact symmetry
    pts += np.random.uniform(-0.002, 0.002, pts.shape)
    
    # Compute safe initial radii to guarantee feasibility
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        for j in range(n):
            if i != j:
                d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d * 0.40  # Safety margin
        
    x0 = np.zeros(3*n)
    x0[0::3] = pts[:,0]
    x0[1::3] = pts[:,1]
    x0[2::3] = radii
    return x0

def solve_lp_radii(centers):
    """Optimally scale radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        bound = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, bound)
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
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def run_packing():
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Extensive multi-start with varied rotations and densities
    configs = []
    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    for ang in angles:
        configs.append(generate_hex_init(N, ang, r_base=0.095))
        configs.append(generate_hex_init(N, ang, r_base=0.105))
        
    for x0 in configs:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 30000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Deflation, Perturbation & LP Refinement to escape local minima
    if best_x is not None:
        for step in range(10):
            x0 = best_x.copy()
            # Shrink radii slightly to free up space for repositioning
            x0[2::3] *= 0.980
            # Perturb centers to explore new topologies
            x0[0::3] += np.random.normal(0, 0.002, N)
            x0[1::3] += np.random.normal(0, 0.002, N)
            
            # Project perturbed variables back to strict bounds
            for i in range(N):
                r = max(0.005, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0-r)
                x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0-r)
                x0[3*i+2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-7 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
                        
                        # LP refinement on optimized centers to snap radii to theoretical max
                        centers_lp = np.column_stack((best_x[0::3], best_x[1::3]))
                        r_lp = solve_lp_radii(centers_lp)
                        if r_lp is not None:
                            best_x[2::3] = r_lp
                            best_sum = np.sum(r_lp)
            except Exception:
                pass
                
    # Fallback if optimization completely fails
    if best_x is None:
        best_x = generate_hex_init(N, 15.0, 0.10)
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
