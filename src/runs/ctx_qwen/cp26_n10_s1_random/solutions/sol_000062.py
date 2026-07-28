# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000010 (state f39c4564) state=2d69ac4d sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def stage1_obj(x, n):
    """Objective for Stage 1: maximize equal radius r (minimize -r)"""
    return -x[-1]

def stage1_cons(x, n):
    """Constraints for Stage 1: boundaries and pairwise non-overlap for equal radius r"""
    cxs = x[0::2]
    cys = x[1::2]
    r = x[-1]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    b_cons = np.concatenate([
        cxs - r,
        1.0 - cxs - r,
        cys - r,
        1.0 - cys - r
    ])
    
    # Pairwise distance constraints: dist(i,j) >= 2r
    diffs = cxs[:, np.newaxis] - cxs[np.newaxis, :]
    diffy = cys[:, np.newaxis] - cys[np.newaxis, :]
    dists = np.sqrt(diffs**2 + diffy**2)
    np.fill_diagonal(dists, np.inf)
    
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p_cons = dists[triu_mask] - 2.0 * r
    
    return np.concatenate([b_cons, p_cons])

def run_packing() -> tuple:
    n = 26
    
    # Generate multiple initial configurations from perturbed hexagonal lattices
    inits = []
    np.random.seed(123)
    for _ in range(6):
        r0 = 0.09 + np.random.rand() * 0.02
        sx = np.random.uniform(-0.05, 0.05)
        sy = np.random.uniform(-0.05, 0.05)
        rot = np.random.uniform(-0.1, 0.1)
        
        pts = []
        y = r0 - 0.1
        while y < 1.1:
            x = r0 - 0.1
            row_idx = int((y - r0) / (np.sqrt(3)*r0))
            offset = (row_idx % 2) * r0
            while x < 1.1:
                pts.append([x + offset, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            
        pts = np.array(pts)
        c = np.array([0.5, 0.5])
        pts = pts - c
        cos_a, sin_a = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = pts + c + np.array([sx, sy])
        
        valid = (pts[:,0] >= 0) & (pts[:,0] <= 1) & (pts[:,1] >= 0) & (pts[:,1] <= 1)
        valid_pts = pts[valid]
        if len(valid_pts) >= n:
            dists = np.sum((valid_pts - 0.5)**2, axis=1)
            idx = np.argsort(dists)[:n]
            inits.append(valid_pts[idx])
        else:
            # Fallback to grid if lattice generation fails
            gx = np.linspace(0.05, 0.95, 6)
            gy = np.linspace(0.05, 0.95, 5)
            grid = np.array([[x,y] for y in gy for x in gx])[:n]
            inits.append(grid)

    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_stg1 = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)]
    
    for centers0 in inits:
        x0 = np.concatenate([centers0.flatten(), [0.08]])
        
        try:
            # Stage 1: Optimize centers for equal radii
            res = minimize(stage1_obj, x0, args=(n,), method='SLSQP', bounds=bounds_stg1, 
                           constraints={'type': 'ineq', 'fun': stage1_cons, 'args': (n,)},
                           options={'maxiter': 2000, 'ftol': 1e-9})
            
            c_opt = res.x[:2*n].reshape(n, 2)
            
            # Stage 2: Solve LP for variable radii given fixed centers
            # Maximize sum(r_i) <=> Minimize -sum(r_i)
            c_obj = -np.ones(n)
            n_constraints = 4*n + n*(n-1)//2
            A_ub = np.zeros((n_constraints, n))
            b_ub = np.zeros(n_constraints)
            
            idx = 0
            for i in range(n):
                # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
                A_ub[idx, i] = 1.0; b_ub[idx] = c_opt[i, 0]; idx += 1
                A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - c_opt[i, 0]; idx += 1
                A_ub[idx, i] = 1.0; b_ub[idx] = c_opt[i, 1]; idx += 1
                A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - c_opt[i, 1]; idx += 1
                
            for i in range(n):
                for j in range(i+1, n):
                    # Pairwise constraints: r_i + r_j <= dist(i,j)
                    dist = np.linalg.norm(c_opt[i] - c_opt[j])
                    A_ub[idx, i] = 1.0; A_ub[idx, j] = 1.0; b_ub[idx] = dist; idx += 1
                    
            bounds_r = [(0.0, None)] * n
            lp_res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
            
            if lp_res.success:
                r_opt = lp_res.x
                s = np.sum(r_opt)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt
                    best_radii = r_opt
        except Exception:
            continue

    # Fallback if optimization fails entirely
    if best_centers is None:
        best_centers = inits[0]
        best_radii = np.full(n, 0.05)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
