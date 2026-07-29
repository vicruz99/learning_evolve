# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000056 (state 0fa800b4) state=dffbd27d sum of radii=2.621719 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def obj(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def con(vars):
    """Computes boundary and non-overlap constraints."""
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    
    c = []
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    i, j = np.triu_indices(N, 1)
    dx = xs[i] - xs[j]
    dy = ys[i] - ys[j]
    dr = rs[i] + rs[j]
    c.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(c)

def get_bounds():
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.extend([(0,1), (0,1), (1e-6, 0.5)])
    return b

def run_packing():
    np.random.seed(42)
    bds = get_bounds()
    cons = {'type': 'ineq', 'fun': con}
    
    best_sol = None
    best_sum = -1.0
    
    starts = []
    
    # 1. Hexagonal lattice initializations (high density baseline)
    for r0 in [0.095, 0.100, 0.105, 0.110]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N:
            x = r0 + (row % 2) * r0
            while x <= 1 - r0 and len(pts) < N:
                pts.append([x, y])
                x += 2 * r0
            y += r0 * np.sqrt(3)
            row += 1
        while len(pts) < N:
            pts.append([0.5, 0.5])
        pts = np.array(pts[:N]) + np.random.normal(0, 0.002, (N, 2))
        pts = np.clip(pts, 0.01, 0.99)
        v = np.zeros(N*3)
        v[0::3] = pts[:,0]
        v[1::3] = pts[:,1]
        v[2::3] = r0 * 0.98
        starts.append(v)
        
    # 2. Random initializations (explores non-lattice optima)
    for _ in range(10):
        pts = np.random.rand(N, 2) * 0.8 + 0.1
        v = np.zeros(N*3)
        v[0::3] = pts[:,0]
        v[1::3] = pts[:,1]
        v[2::3] = 0.08
        starts.append(v)
        
    # 3. Grid initializations (structured baseline)
    for scale in [0.9, 1.0, 1.1]:
        xg = np.linspace(0.1, 0.9, 5)
        yg = np.linspace(0.1, 0.9, 5)
        pts = np.array([[x,y] for x in xg for y in yg])
        pts = np.vstack([pts, [0.5, 0.5]])[:N]
        pts = (pts - 0.5) * scale + 0.5
        pts += np.random.normal(0, 0.005, pts.shape)
        pts = np.clip(pts, 0.01, 0.99)
        v = np.zeros(N*3)
        v[0::3] = pts[:,0]
        v[1::3] = pts[:,1]
        v[2::3] = 0.08 * scale
        starts.append(v)
        
    # 4. Corner-focused initializations (utilizes boundary efficiently)
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    for _ in range(5):
        pts = corners.copy()
        rem = np.random.rand(N-4, 2) * 0.6 + 0.2
        pts = np.vstack([pts, rem])
        v = np.zeros(N*3)
        v[0::3] = pts[:,0]
        v[1::3] = pts[:,1]
        v[2::3] = 0.09
        starts.append(v)
        
    # Phase 1: Multi-start optimization
    for s in starts:
        try:
            res = minimize(obj, s, method='SLSQP', bounds=bds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if np.all(con(res.x) >= -1e-7):
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_sol = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative radius growth & refinement
    # Gradually increase radii while re-optimizing centers to climb the objective
    if best_sol is not None:
        curr = best_sol.copy()
        for step in range(30):
            curr += np.random.normal(0, 0.0005, curr.shape)
            curr[2::3] *= 1.002
            curr[2::3] = np.clip(curr[2::3], 1e-6, 0.5)
            try:
                res = minimize(obj, curr, method='SLSQP', bounds=bds, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
                if np.all(con(res.x) >= -1e-7):
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_sol = res.x.copy()
                        curr = best_sol.copy()
            except Exception:
                curr = best_sol.copy()
                
    # Phase 3: Strict feasibility repair
    if best_sol is not None:
        for _ in range(20):
            if np.all(con(best_sol) >= -1e-9):
                break
            best_sol[2::3] *= 0.9995
            
        centers = best_sol.reshape(N, 3)[:, :2]
        radii = best_sol.reshape(N, 3)[:, 2]
        radii = np.maximum(radii, 0.0)
        return centers, radii, float(np.sum(radii))
        
    # Fallback (should not be reached with valid optimization)
    c = np.random.rand(N, 2) * 0.6 + 0.2
    r = np.full(N, 0.06)
    return c, r, float(np.sum(r))
