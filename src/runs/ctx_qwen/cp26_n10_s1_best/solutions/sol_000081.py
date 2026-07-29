# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000061 (state 63a33892) state=a0d6c193 sum of radii=2.626678 correctness=1.0
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
    """Returns all inequality constraints g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Overlap constraints using squared distances
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
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

def generate_hex_init(r0, angle):
    """Generate a hexagonal lattice initialization with optional rotation."""
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
        if y > 1.0 + r0:
            break
            
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (1, 2))])
        
    return pts[:N]

def generate_force_init(seed):
    """Generate initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (N, 2))
    
    for _ in range(600):
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[j] - centers[i]
                dist = np.linalg.norm(diff)
                if dist < 0.28 and dist > 1e-6:
                    f = 0.015 / (dist**2 + 0.001)
                    forces[i] -= f * diff
                    forces[j] += f * diff
                    
            for dim in range(2):
                if centers[i, dim] < 0.12:
                    forces[i, dim] += 0.025
                elif centers[i, dim] > 0.88:
                    forces[i, dim] -= 0.025
                    
        centers += forces * 0.05
        centers = np.clip(centers, 0.05, 0.95)
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattices with varied densities and rotations
    for r0 in np.linspace(0.075, 0.105, 5):
        for ang in np.linspace(0.0, np.pi / 6, 9):
            inits.append(generate_hex_init(r0, ang))
            
    # Force-directed layouts
    for s in range(8):
        inits.append(generate_force_init(s))
        
    # Grid-like layouts (perturbed)
    for _ in range(5):
        pts = np.zeros((N, 2))
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx < N:
                    pts[idx, 0] = 0.08 + j * 0.18
                    pts[idx, 1] = 0.08 + i * 0.16
                    idx += 1
        pts += np.random.uniform(-0.01, 0.01, pts.shape)
        inits.append(pts)
        
    # Phase 2: Multi-start optimization with LP refinement
    for c_init in inits:
        # Use LP to find optimal radii for these centers
        r_lp = solve_lp_radii(c_init)
        if r_lp is None:
            r_lp = np.full(N, 0.05)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_lp
        
        # Perturb slightly to break exact symmetries
        x0 += np.random.normal(0, 1e-4, x0.shape)
        
        # Project to strict bounds
        for i in range(N):
            r = max(1e-7, x0[3 * i + 2])
            x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
            x0[3 * i + 2] = r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            if not np.isnan(res.fun):
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
                    
                    # LP refinement on optimized centers
                    c_new = best_x.reshape(N, 3)[:, :2]
                    r_lp = solve_lp_radii(c_new)
                    if r_lp is not None and np.sum(r_lp) > best_sum:
                        best_x[2::3] = r_lp
                        best_sum = np.sum(r_lp)
        except Exception:
            continue
            
    # Phase 3: Iterative deflation & refinement to escape local minima
    if best_x is not None:
        for step in range(20):
            noise_scale = 0.003 / (step + 1)
            x_pert = best_x + np.random.normal(0, noise_scale, best_x.shape)
            
            # Shrink radii slightly to guarantee feasibility for optimizer
            r_pert = np.maximum(x_pert[2::3] * 0.985, 1e-5)
            x_pert[2::3] = r_pert
            
            for i in range(N):
                r = r_pert[i]
                x_pert[3 * i] = np.clip(x_pert[3 * i], r, 1.0 - r)
                x_pert[3 * i + 1] = np.clip(x_pert[3 * i + 1], r, 1.0 - r)
                
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                
                if not np.isnan(res.fun):
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-7 and -res.fun > best_sum:
                        best_sum = -res.fun
                        best_x = res.x.copy()
                        
                        c_new = best_x.reshape(N, 3)[:, :2]
                        r_lp = solve_lp_radii(c_new)
                        if r_lp is not None and np.sum(r_lp) > best_sum:
                            best_x[2::3] = r_lp
                            best_sum = np.sum(r_lp)
            except Exception:
                pass
                
    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[2::3] = 0.06
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x[2::3]
    
    # Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-12 or centers[i, 0] > 1.0 - radii[i] + 1e-12 or \
               centers[i, 1] < radii[i] - 1e-12 or centers[i, 1] > 1.0 - radii[i] + 1e-12:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Minimal shrinkage to recover validity
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
