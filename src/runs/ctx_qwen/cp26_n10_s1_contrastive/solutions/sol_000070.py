# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=09794d95 sum of radii=2.623001 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def compute_constraints(x):
    """Compute all inequality constraints: boundary and separation."""
    N = 26
    C = x.reshape(N, 3)
    xc = C[:, 0]
    yc = C[:, 1]
    r = C[:, 2]
    
    c = []
    # Boundary constraints: circle inside [0,1]^2
    c.append(xc - r)
    c.append(1.0 - xc - r)
    c.append(yc - r)
    c.append(1.0 - yc - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = xc[i_idx] - xc[j_idx]
    dy = yc[i_idx] - yc[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    c.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c)

def get_max_radii(centers):
    """Compute strictly feasible initial radii for given centers."""
    N = centers.shape[0]
    radii = np.zeros(N)
    for i in range(N):
        d_bound = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        d_min = 1.0
        for j in range(N):
            if i != j:
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                if d < d_min:
                    d_min = d
        # Use 0.45 factor to ensure strict feasibility for SLSQP start
        radii[i] = min(d_bound, 0.5 * d_min) * 0.45
    return radii

def run_packing():
    N = 26
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = -np.inf
    best_x = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattice variations
    for seed in range(4):
        np.random.seed(seed)
        pts = []
        r_est = 0.09
        y = r_est
        row = 0
        while len(pts) < N:
            x_off = r_est if row % 2 == 0 else 2*r_est
            x = x_off
            while x <= 1.0 - r_est and len(pts) < N:
                pts.append([x, y])
                x += 2*r_est
            y += np.sqrt(3)*r_est
            row += 1
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.03, 0.03, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)
        
    # 2. Grid variations
    for seed in range(4):
        np.random.seed(seed+10)
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i*0.2, 0.1 + j*0.2])
        pts.append([0.5, 0.5])
        pts = np.array(pts)
        pts += np.random.uniform(-0.04, 0.04, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)
        
    # 3. Random uniform
    for seed in range(6):
        np.random.seed(seed+20)
        inits.append(np.random.uniform(0.1, 0.9, (N, 2)))
        
    # 4. Force-directed layout
    for seed in range(3):
        np.random.seed(seed+30)
        pts = np.random.rand(N, 2) * 0.8 + 0.1
        for _ in range(150):
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, 1e-5)
            inv_dists = 1.0 / dists
            forces = np.sum(diffs * inv_dists[..., np.newaxis], axis=1)
            pts += forces * 0.005
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)

    # Phase 1: Broad search from diverse starts
    for pts in inits:
        r_safe = get_max_radii(pts)
        x0 = np.zeros(3*N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_safe
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-12})
            if res.success:
                cons_val = compute_constraints(res.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative perturbation refinement to escape local minima
    if best_x is not None:
        for iter_round in range(3):
            for k in range(6):
                x_pert = best_x + np.random.uniform(-0.003, 0.003, 3*N)
                x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
                x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
                x_pert[2::3] = np.maximum(x_pert[2::3], 1e-6)
                
                try:
                    res = minimize(compute_objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 2000, 'ftol': 1e-12})
                    if res.success:
                        cons_val = compute_constraints(res.x)
                        if np.min(cons_val) >= -1e-9:
                            curr_sum = -res.fun
                            if curr_sum > best_sum:
                                best_sum = curr_sum
                                best_x = res.x.copy()
                except Exception:
                    pass
                    
        # Phase 3: Final high-precision polish
        try:
            res_final = minimize(compute_objective, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                                 options={'maxiter': 6000, 'ftol': 1e-14})
            if res_final.success:
                cons_val = compute_constraints(res_final.x)
                if np.min(cons_val) >= -1e-9:
                    best_x = res_final.x
                    best_sum = -res_final.fun
        except Exception:
            pass

    # Fallback valid configuration
    if best_x is None:
        best_x = np.zeros(3*N)
        best_x[0::3] = np.linspace(0.1, 0.9, N)
        best_x[1::3] = np.linspace(0.1, 0.9, N)
        best_x[2::3] = 0.01
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
