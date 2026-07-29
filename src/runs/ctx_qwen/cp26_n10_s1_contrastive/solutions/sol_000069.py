# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=e1ac005d sum of radii=2.624758 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def get_constraints(x):
    """Compute all inequality constraints: boundary containment and pairwise separation."""
    n = 26
    c = x.reshape(n, 3)
    xc, yc, r = c[:, 0], c[:, 1], c[:, 2]
    
    con = []
    # Boundary constraints: circle must be inside [0,1]^2
    con.append(xc - r)
    con.append(1.0 - xc - r)
    con.append(yc - r)
    con.append(1.0 - yc - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(n, 1)
    dx = xc[i_idx] - xc[j_idx]
    dy = yc[i_idx] - yc[j_idx]
    rs = r[i_idx] + r[j_idx]
    con.append(dx*dx + dy*dy - rs*rs)
    
    return np.concatenate(con)

def generate_force_init(seed):
    """Generate a well-spread initial configuration using force-directed repulsion."""
    n = 26
    np.random.seed(seed)
    pts = np.random.rand(n, 2) * 0.7 + 0.15
    
    for t in range(500):
        # Vectorized repulsion calculation
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists[dists < 1e-7] = 1e-7
        f_mag = 1.0 / (dists**2 + 0.005)
        forces = np.sum(diff * f_mag[:, :, None], axis=1)
        
        # Wall repulsion: push points away from boundaries
        forces += 5.0 * np.where(pts < 0.2, 1.0, 0.0)
        forces += -5.0 * np.where(pts > 0.8, 1.0, 0.0)
        
        # Adaptive step size
        step = 0.03 / (1.0 + t/100.0)
        pts += step * forces
        pts = np.clip(pts, 0.01, 0.99)
        
    return pts

def compute_feasible_radii(pts):
    """Compute strictly feasible radii for a given set of centers."""
    n = pts.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        d_wall = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        d_min = 1.0
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < d_min:
                    d_min = d
        radii[i] = 0.95 * min(d_wall, 0.5 * d_min)
    return radii

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_x = None
    best_sum = -np.inf
    
    # Phase 1: Diverse initializations
    for seed in range(25):
        if seed < 8:
            # Hexagonal lattice initialization with perturbation
            pts = []
            r_est = 0.095
            y = r_est
            row = 0
            while len(pts) < n:
                shift = (row % 2) * r_est
                x = r_est + shift
                while x <= 1.0 - r_est and len(pts) < n:
                    pts.append([x, y])
                    x += 2.0 * r_est
                y += np.sqrt(3.0) * r_est
                row += 1
            pts = np.array(pts[:n])
            pts += np.random.randn(n, 2) * 0.02
            pts = np.clip(pts, 0.02, 0.98)
        else:
            # Force-directed layout initialization
            pts = generate_force_init(seed)
            
        radii = compute_feasible_radii(pts)
        
        x0 = np.zeros(3 * n)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = radii
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
            if res.success:
                con_val = get_constraints(res.x)
                if np.min(con_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement on best solution
    if best_x is not None:
        for _ in range(15):
            x_pert = best_x.copy()
            x_pert += np.random.randn(3 * n) * 0.004
            x_pert[0::3] = np.clip(x_pert[0::3], 0.001, 0.999)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.001, 0.999)
            x_pert[2::3] = np.clip(x_pert[2::3], 1e-6, 0.499)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
                if res.success:
                    con_val = get_constraints(res.x)
                    if np.min(con_val) >= -1e-9:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_x = res.x.copy()
            except Exception:
                continue
                
    # Phase 3: High-precision polish
    if best_x is not None:
        try:
            res = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14})
            if res.success:
                con_val = get_constraints(res.x)
                if np.min(con_val) >= -1e-10:
                    best_x = res.x
                    best_sum = -res.fun
        except Exception:
            pass
            
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
