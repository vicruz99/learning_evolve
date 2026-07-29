# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000072 (state e356f834) state=0bf82bcb sum of radii=2.619761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Compute all boundary and non-overlap constraints as a vector >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([cx - cr, 1.0 - cx - cr, cy - cr, 1.0 - cy - cr])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.tril_indices(N, -1)
    dx = cx[i_idx] - cx[j_idx]
    dy = cy[i_idx] - cy[j_idx]
    dr = cr[i_idx] + cr[j_idx]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N

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
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return None

def make_hex_init(r0, angle, seed):
    """Generate a rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 5:
        x_start = r0 if row % 2 == 0 else 2.0 * r0
        x = x_start
        while x <= 1.0 - r0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += math.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N + 5])
    
    # Rotate around center
    if angle != 0.0:
        c = np.array([0.5, 0.5])
        ca, sa = np.cos(angle), np.sin(angle)
        pts = ((pts - c) @ np.array([[ca, -sa], [sa, ca]]) + c)
        
    # Filter points strictly inside
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & \
           (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    
    # Pad if necessary
    while len(pts) < N:
        pts = np.vstack([pts, [np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)]])
    pts = pts[:N]
    
    # Compute safe radii via LP for these centers
    radii = solve_lp_radii(pts)
    if radii is None:
        radii = np.full(N, r0 * 0.95)
    else:
        radii = radii * 0.99  # Slight shrink for strict feasibility in squared constraints
        
    x0 = np.zeros(3 * N)
    x0[0::3] = pts[:, 0]
    x0[1::3] = pts[:, 1]
    x0[2::3] = radii
    return x0

def make_force_init(seed):
    """Generate initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    n = N
    cx = np.random.uniform(0.15, 0.85, n)
    cy = np.random.uniform(0.15, 0.85, n)
    cr = np.full(n, 0.08)
    
    for step in range(800):
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        # Mask out self-distances and far pairs
        valid = (d < 0.25) & (d > 1e-6)
        force_mag = np.zeros_like(d2)
        force_mag[valid] = 0.02 / (d2[valid] + 0.0001)
        
        fx = np.sum(force_mag * dx, axis=1)
        fy = np.sum(force_mag * dy, axis=1)
        
        # Wall repulsion
        w = 0.05
        fx += np.where(cx < cr + 0.05, w, 0.0) - np.where(cx > 1.0 - cr - 0.05, w, 0.0)
        fy += np.where(cy < cr + 0.05, w, 0.0) - np.where(cy > 1.0 - cr - 0.05, w, 0.0)
        
        alpha = 0.03 * (1.0 - step / 800.0)
        cx += fx * alpha
        cy += fy * alpha
        
        cx = np.clip(cx, 0.02, 0.98)
        cy = np.clip(cy, 0.02, 0.98)
        
    centers = np.column_stack((cx, cy))
    radii = solve_lp_radii(centers)
    if radii is None:
        radii = np.full(N, 0.06)
    else:
        radii = radii * 0.99
        
    x0 = np.zeros(3 * N)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def run_packing():
    bnds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse multi-start
    inits = []
    # Hexagonal lattice with varied density and rotation
    for s in range(25):
        r0 = 0.085 + s * 0.0015
        inits.append(make_hex_init(r0, angle=s * 0.05, seed=s * 100))
        
    # Force-directed layouts
    for s in range(10):
        inits.append(make_force_init(seed=s))
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            curr_sum = -res.fun
            c_vals = constraints(res.x)
            if np.min(c_vals) >= -1e-8 and curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative Deflation-Perturbation Refinement
    if best_x is not None:
        for rnd in range(15):
            x0 = best_x.copy()
            # Deflate radii to free space
            x0[2::3] *= 0.97
            
            noise = 0.002 * (0.8 ** rnd)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Project to bounds
            for i in range(N):
                r = max(0.01, x0[3*i + 2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0 - r)
                x0[3*i + 1] = np.clip(x0[3*i + 1], r, 1.0 - r)
                x0[3*i + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                curr_sum = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-8 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
            except Exception:
                pass
                
            # LP refinement on best centers found so far
            if best_x is not None:
                centers_cur = np.column_stack((best_x[0::3], best_x[1::3]))
                lp_r = solve_lp_radii(centers_cur)
                if lp_r is not None:
                    lp_r = lp_r * 0.999
                    best_x[2::3] = lp_r
                    curr_lp_sum = np.sum(lp_r)
                    if curr_lp_sum > best_sum:
                        best_sum = curr_lp_sum
                        
    # Fallback
    if best_x is None:
        best_x = make_hex_init(0.09, 0.0, 0)
        best_sum = -objective(best_x)
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Phase 3: Strict validation and numerical repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if (radii[i] < 0 or 
                centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9 or 
                centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9):
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            break
            
        # Minimal shrinkage to guarantee strict compliance
        radii *= 0.999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
