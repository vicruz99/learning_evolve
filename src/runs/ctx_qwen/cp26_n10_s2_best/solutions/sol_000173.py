# sol_000173 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000150 (state 86f9e7dc) state=c294b971 sum of radii=1.663528 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def solve_radii_lp(centers):
    """
    Solves the radius subproblem exactly using Linear Programming.
    Maximize sum(r_i) subject to r_i <= dist_to_boundary and r_i + r_j <= dist_ij.
    """
    c_obj = -np.ones(N)
    A_ub = np.zeros((4*N + NUM_PAIRS, N))
    b_ub = np.zeros(4*N + NUM_PAIRS)
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(N):
        b_ub[4*i] = centers[i, 0]
        b_ub[4*i+1] = 1.0 - centers[i, 0]
        b_ub[4*i+2] = centers[i, 1]
        b_ub[4*i+3] = 1.0 - centers[i, 1]
        A_ub[4*i : 4*i+4, i] = 1.0
        
    # Pairwise constraints: r_i + r_j <= dist_ij
    dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
    dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
    dists = np.hypot(dx, dy)
    
    for k in range(NUM_PAIRS):
        row = 4*N + k
        A_ub[row, PAIR_I[k]] = 1.0
        A_ub[row, PAIR_J[k]] = 1.0
        b_ub[row] = dists[k]
        
    bounds = [(0.0, None)] * N
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 0.01)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Uses hybrid LP+NLP strategy for superior convergence.
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    inits = []
    
    # 1. Rotated Hexagonal Lattices
    for r0 in np.linspace(0.08, 0.11, 9):
        for ang in np.linspace(-0.2, 0.2, 5):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 5:
                xs = r0 if row%2==0 else 2*r0
                x = xs
                while x <= 1.0-r0 and len(pts) < N+5:
                    pts.append([x, y])
                    x += 2*r0
                y += np.sqrt(3)*r0
                row += 1
            pts = np.array(pts[:N])
            if ang != 0:
                c, s = np.cos(ang), np.sin(ang)
                pts = (pts - 0.5) @ [[c, -s], [s, c]] + 0.5
            pts += np.random.uniform(-0.005, 0.005, pts.shape)
            pts = np.clip(pts, 0.02, 0.98)
            inits.append(pts)
            
    # 2. Force-relaxed random configurations
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        for _ in range(100):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i]-pts[j])
                    if d < 0.25 and d > 1e-4:
                        f = (0.25-d)/d * 0.1
                        forces[i] += f*(pts[i]-pts[j])
                        forces[j] -= f*(pts[i]-pts[j])
            pts += forces * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    # Phase 1: Multi-start with LP radius initialization
    for centers in inits:
        r_init = solve_radii_lp(centers)
        v0 = np.concatenate([centers[:,0], centers[:,1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            
            # Extract optimized centers and solve LP for exact optimal radii
            opt_centers = res.x[:2*N].reshape(N, 2)
            opt_radii = solve_radii_lp(opt_centers)
            v_cand = np.concatenate([opt_centers[:,0], opt_centers[:,1], opt_radii])
            
            if np.min(constraints(v_cand)) >= -1e-6:
                s = np.sum(opt_radii)
                if s > best_sum:
                    best_sum = s
                    best_v = v_cand.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation & LP-refined optimization to escape local minima
    if best_v is not None:
        curr_v = best_v.copy()
        for step in range(30):
            np.random.seed(step + 100)
            v_p = curr_v.copy()
            noise = 0.005 * (1.0 - step/30.0)
            v_p[:2*N] += np.random.uniform(-noise, noise, 2*N)
            v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
            
            # Immediately compute optimal radii for perturbed centers
            centers_p = v_p[:2*N].reshape(N, 2)
            r_p = solve_radii_lp(centers_p)
            v_p[2*N:] = r_p
            
            try:
                res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                               
                opt_centers = res.x[:2*N].reshape(N, 2)
                opt_radii = solve_radii_lp(opt_centers)
                v_cand = np.concatenate([opt_centers[:,0], opt_centers[:,1], opt_radii])
                
                if np.min(constraints(v_cand)) >= -1e-6:
                    s = np.sum(opt_radii)
                    if s > best_sum:
                        best_sum = s
                        best_v = v_cand.copy()
                        curr_v = best_v.copy()
            except Exception:
                pass
                
    # Phase 3: Strict Post-Processing for Validator Compliance
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce pairwise non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-10
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
