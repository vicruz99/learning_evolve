# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000052 (state 0d4d18bd) state=1cdab01a sum of radii=2.627905 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRI = np.triu_indices(N, 1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Vectorized inequality constraints g(x) >= 0."""
    xc = x[0::3]
    yc = x[1::3]
    r = x[2::3]
    
    # Preallocate constraint array
    n_cons = 4 * N + N * (N - 1) // 2
    c = np.empty(n_cons)
    
    # Boundary constraints
    c[:N] = xc - r
    c[N:2*N] = 1.0 - xc - r
    c[2*N:3*N] = yc - r
    c[3*N:4*N] = 1.0 - yc - r
    
    # Overlap constraints (squared form for smoothness)
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    dr = r[:, None] + r[None, :]
    
    c[4*N:] = dx[TRI]**2 + dy[TRI]**2 - dr[TRI]**2
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-6, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return b

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with rotations
    for seed in range(15):
        np.random.seed(seed)
        r0 = 0.085 + np.random.uniform(-0.005, 0.005)
        centers = []
        y = r0
        row = 0
        while len(centers) < N:
            x = r0 if row % 2 == 0 else 2 * r0
            while len(centers) < N and x + r0 <= 1.0:
                centers.append([x + np.random.uniform(-0.002, 0.002), 
                                y + np.random.uniform(-0.002, 0.002)])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        while len(centers) < N:
            centers.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        centers = np.array(centers[:N])
        
        # Rotate slightly to break symmetry
        angle = np.random.uniform(-0.3, 0.3)
        c_pts = np.array([0.5, 0.5])
        R = np.array([[np.cos(angle), -np.sin(angle)], 
                       [np.sin(angle), np.cos(angle)]])
        centers = (centers - c_pts) @ R.T + c_pts
        
        x0 = np.zeros(3 * N)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = r0
        inits.append(x0)
        
    # 2. Force-relaxed random points
    for seed in range(20):
        np.random.seed(seed + 100)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(300):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = pts[j] - pts[i]
                    dist = np.linalg.norm(d)
                    if dist < 0.28 and dist > 1e-6:
                        f = 0.008 / (dist**2 + 0.001)
                        forces[i] -= f * d
                        forces[j] += f * d
            # Boundary repulsion
            for i in range(N):
                if pts[i, 0] < 0.12: forces[i, 0] += 0.02
                elif pts[i, 0] > 0.88: forces[i, 0] -= 0.02
                if pts[i, 1] < 0.12: forces[i, 1] += 0.02
                elif pts[i, 1] > 0.88: forces[i, 1] -= 0.02
                
            pts += forces * 0.05
            pts = np.clip(pts, 0.05, 0.95)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = 0.065
        inits.append(x0)
        
    # 3. Structured grid starts
    for shift in [0.0, 0.05, 0.1]:
        x0 = np.zeros(3 * N)
        idx = 0
        y = 0.1 + shift
        while idx < N:
            x = 0.1 + shift
            while idx < N and x <= 0.9 - shift:
                x0[3*idx] = x + np.random.uniform(-0.01, 0.01)
                x0[3*idx+1] = y + np.random.uniform(-0.01, 0.01)
                x0[3*idx+2] = 0.08
                idx += 1
                x += 0.18
            y += 0.16
        inits.append(x0)
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -np.inf
    best_x = None
    
    inits = generate_inits()
    
    # Primary optimization phase
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun) and -res.fun > best_val:
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7:
                    best_val = -res.fun
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Deflation-Refinement phase to escape local minima
    if best_x is not None:
        for it in range(15):
            shrink = 0.992
            x_pert = best_x.copy()
            x_pert[2::3] *= shrink
            x_pert[0::3] += np.random.normal(0, 0.0015, N)
            x_pert[1::3] += np.random.normal(0, 0.0015, N)
            
            # Project to feasible bounds
            r_p = np.maximum(x_pert[2::3], 1e-5)
            x_pert[0::3] = np.clip(x_pert[0::3], r_p, 1.0 - r_p)
            x_pert[1::3] = np.clip(x_pert[1::3], r_p, 1.0 - r_p)
            x_pert[2::3] = r_p
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun) and -res.fun > best_val:
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-7:
                        best_val = -res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Fallback initialization (should rarely be needed)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[2::3] = 0.06
        best_x[0::3] = np.tile(np.linspace(0.15, 0.85, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.15, 0.85, 6), 5)[:N]
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Strict validity repair against 1e-12 tolerance
    for _ in range(100):
        ok = True
        if np.any(radii < 0): ok = False
        if ok:
            if np.any(centers[:,0] - radii < -1e-9) or np.any(centers[:,0] + radii > 1.0 + 1e-9): ok = False
        if ok:
            if np.any(centers[:,1] - radii < -1e-9) or np.any(centers[:,1] + radii > 1.0 + 1e-9): ok = False
        if ok:
            dx = centers[:,0, None] - centers[:,0]
            dy = centers[:,1, None] - centers[:,1]
            d = np.sqrt(dx**2 + dy**2)
            r_sum = radii[:, None] + radii[None, :]
            if np.any(d[TRI] < r_sum[TRI] - 1e-9): ok = False
            
        if ok:
            break
        radii *= 0.9995
        
    return centers, radii, float(np.sum(radii))
