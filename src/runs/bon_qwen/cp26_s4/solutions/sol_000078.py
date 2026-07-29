# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1daf7277) state=416cce81 sum of radii=2.217228 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing():
    n = 26
    # Initialize centers in a staggered hexagonal pattern
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(6):
        y = 0.12 + row * 0.15
        cols = 5 if row % 2 == 0 else 4
        x_start = 0.12 + (0 if row % 2 == 0 else 0.085)
        for col in range(cols):
            if idx < n:
                centers[idx, 0] = x_start + col * 0.17
                centers[idx, 1] = y
                idx += 1
                
    alpha_init = 0.04
    n_iter = 900
    
    for it in range(n_iter):
        # Decaying step size
        alpha = alpha_init * (1.0 - it / n_iter) ** 0.5
        if alpha < 1e-7:
            alpha = 1e-7
            
        # Compute boundary limits
        b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                       np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # LP setup: Maximize sum(r) => Minimize -sum(r)
        c_obj = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= b_i
        for i in range(n):
            A_ub.append(np.eye(n)[i])
            b_ub.append(b[i])
            
        # Pairwise constraints: r_i + r_j <= d_ij
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Solve LP
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if not res.success:
            break
            
        radii = res.x
        
        # Extract dual variables (marginals)
        try:
            marginals = np.asarray(res.marginals)
        except:
            marginals = np.zeros(len(b_ub))
            
        forces = np.zeros_like(centers)
        
        # Boundary forces from duals
        for i in range(n):
            m = marginals[i]
            if m > 1e-9:
                x, y = centers[i]
                # Subgradient of min(x, 1-x, y, 1-y)
                if abs(x - b[i]) < 1e-7:
                    forces[i, 0] += m
                elif abs((1.0 - x) - b[i]) < 1e-7:
                    forces[i, 0] -= m
                if abs(y - b[i]) < 1e-7:
                    forces[i, 1] += m
                elif abs((1.0 - y) - b[i]) < 1e-7:
                    forces[i, 1] -= m
                    
        # Pairwise repulsive forces from duals
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                lam = marginals[n + idx]
                if lam > 1e-9:
                    d = dists[i, j]
                    if d > 1e-9:
                        fx = lam * (centers[i, 0] - centers[j, 0]) / d
                        fy = lam * (centers[i, 1] - centers[j, 1]) / d
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                idx += 1
                
        # Update centers
        centers += alpha * forces
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
    # Final LP to compute exact optimal radii for converged centers
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    A_ub = np.eye(n)
    b_ub = b
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub = np.vstack([A_ub, row])
            b_ub = np.append(b_ub, dists[i, j])
            
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    radii = res.x
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
