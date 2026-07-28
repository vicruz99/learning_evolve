# sol_000151 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000123 (state 90e3970d) state=e47663df sum of radii=0.145591 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def obj_joint(v):
    """Objective: minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(v[2*N:])

def con_joint(v):
    """Inequality constraints >= 0 for valid packing"""
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles inside [0,1]x[0,1]
    c_boundary = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    cx_mat = cx[:, None] - cx[None, :]
    cy_mat = cy[:, None] - cy[None, :]
    d2 = cx_mat**2 + cy_mat**2
    r_sum = r[:, None] + r[None, :]
    
    triu = np.triu_indices(N, k=1)
    c_pair = d2[triu] - r_sum[triu]**2
    
    return np.concatenate([c_boundary, c_pair])

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers"""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A = []
    b = []
    
    # Boundary constraints: r_i <= dist to walls
    for i in range(n):
        x, y = centers[i]
        for lim in (x, 1.0 - x, y, 1.0 - y):
            row = np.zeros(n)
            row[i] = 1.0
            A.append(row)
            b.append(lim)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A.append(row)
            b.append(d)
            
    A = np.array(A)
    b = np.array(b)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def force_expand(centers, steps=1200):
    """Force-directed simulation to push circles into a dense configuration"""
    n = centers.shape[0]
    centers = centers.copy()
    r = np.full(n, 0.04)
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    triu_i, triu_j = np.triu_indices(n, k=1)
    
    for _ in range(steps):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            ri = r[i]
            if cx[i] < ri: fx[i] += (ri - cx[i]) * 100.0
            if cx[i] > 1.0 - ri: fx[i] -= (cx[i] - (1.0 - ri)) * 100.0
            if cy[i] < ri: fy[i] += (ri - cy[i]) * 100.0
            if cy[i] > 1.0 - ri: fy[i] -= (cy[i] - (1.0 - ri)) * 100.0
            
        # Pairwise repulsion (vectorized)
        dx = cx[triu_i] - cx[triu_j]
        dy = cy[triu_i] - cy[triu_j]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        d[d < 1e-8] = 1e-8
        pen = np.maximum(0.0, r[triu_i] + r[triu_j] - d)
        f_mag = pen / d
        
        fx[triu_i] += f_mag * dx * 50.0
        fy[triu_i] += f_mag * dy * 50.0
        fx[triu_j] -= f_mag * dx * 50.0
        fy[triu_j] -= f_mag * dy * 50.0
        
        # Update positions
        cx += fx * 0.01
        cy += fy * 0.01
        cx = np.clip(cx, 1e-4, 0.9999)
        cy = np.clip(cy, 1e-4, 0.9999)
        
        # Gradually grow radii to explore larger packings
        r += 1e-5
        
    return np.column_stack((cx, cy)), r

def run_packing():
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # --- Phase 1: Generate Diverse Initial Configurations ---
    inits = []
    
    # Hexagonal lattices at varying densities
    for dens in [0.14, 0.16, 0.18, 0.20]:
        pts = []
        y = dens
        row = 0
        while len(pts) < N + 5:
            shift = dens / 2.0 if row % 2 == 1 else 0.0
            x = dens + shift
            while x < 1.0 - dens:
                pts.append([x, y])
                x += dens
            y += dens * np.sqrt(3) / 2.0
            row += 1
        pts = np.array(pts[:N])
        inits.append(pts)
        
    # Rotated hexagonal lattices
    base = inits[0].copy()
    for rot in [0.1, 0.2, 0.3, 0.4]:
        cos_t, sin_t = np.cos(rot), np.sin(rot)
        rot_pts = base @ np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rot_pts -= rot_pts.min(axis=0)
        rot_pts /= rot_pts.max(axis=0)
        rot_pts = rot_pts * 0.9 + 0.05
        inits.append(rot_pts)
        
    # Regular grid and grid+center
    gx = np.linspace(0.12, 0.88, 5)
    gy = np.linspace(0.12, 0.88, 5)
    grid = np.array([(x, y) for y in gy for x in gx])
    inits.append(grid)
    inits.append(np.vstack([grid, [0.5, 0.5]]))
    
    # Random dense start
    inits.append(rng.uniform(0.15, 0.85, (N, 2)))
    
    bounds_opt = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    
    # --- Phase 2: Explore Configurations via Force + LP + SLSQP ---
    for init_c in inits:
        # Force expansion to find tight center layout
        fc, _ = force_expand(init_c, steps=1000)
        
        # LP to get optimal radii for these centers
        rp, sp = solve_lp(fc)
        if rp is not None and sp > best_sum:
            best_sum = sp
            best_centers = fc.copy()
            best_radii = rp.copy()
            
        # Joint SLSQP refinement from force-expanded centers
        x0 = np.concatenate([fc.flatten(), np.full(N, 0.08)])
        try:
            res = minimize(obj_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints={'type': 'ineq', 'fun': con_joint},
                           options={'maxiter': 4000, 'ftol': 1e-13})
            if res.success:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_centers = res.x[:2*N].reshape(N, 2)
                    best_radii = res.x[2*N:]
        except Exception:
            pass
            
    # --- Phase 3: Perturbation & Refinement of Best Found ---
    if best_centers is not None:
        for _ in range(8):
            x0_p = np.concatenate([best_centers.flatten(), best_radii])
            x0_p[:2*N] += rng.uniform(-0.015, 0.015, 2*N)
            x0_p[:2*N] = np.clip(x0_p[:2*N], 0.05, 0.95)
            try:
                res = minimize(obj_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                               constraints={'type': 'ineq', 'fun': con_joint},
                               options={'maxiter': 4000, 'ftol': 1e-13})
                if res.success:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_centers = res.x[:2*N].reshape(N, 2)
                        best_radii = res.x[2*N:]
            except Exception:
                pass
                
    # --- Phase 4: Strict Safety Scaling ---
    scale = 1.0
    for i in range(N):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((best_centers[i,0] - best_centers[j,0])**2 + (best_centers[i,1] - best_centers[j,1])**2)
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
