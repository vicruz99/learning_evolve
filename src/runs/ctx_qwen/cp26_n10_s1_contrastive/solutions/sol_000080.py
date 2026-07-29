# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state d15e4e7a) state=b3333e60 sum of radii=2.630831 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
_BOUNDS = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def constraint_func(v):
    """
    Computes inequality constraints: boundary containment and pairwise separation.
    All constraints are formulated as g(v) >= 0.
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    # Upper triangular mask to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c.append(dist2[mask] - rs[mask]**2)
    
    return np.concatenate(c)

def get_safe_radii(centers):
    """Computes strictly feasible radii for given centers."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        d_wall = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        d_min = np.inf
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                if d < d_min:
                    d_min = d
        # 0.98 factor ensures strict feasibility for SLSQP start
        r[i] = 0.98 * min(d_wall, 0.5 * d_min)
    return r

def run_packing():
    best_sum = -np.inf
    best_v = None
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Phase 1: Diverse restarts to explore global landscape
    for seed in range(50):
        np.random.seed(seed)
        centers = np.zeros((N, 2))
        
        if seed % 3 == 0:
            # Hexagonal lattice inspiration
            pts = []
            r_est = 0.09
            y = r_est
            row = 0
            while len(pts) < N:
                x_off = (row % 2) * r_est
                x = r_est + x_off
                while x <= 1.0 - r_est and len(pts) < N:
                    pts.append([x, y])
                    x += 2.0 * r_est
                y += np.sqrt(3.0) * r_est
                row += 1
            centers = np.array(pts[:N])
        elif seed % 3 == 1:
            # Square grid with central extra circle
            idx = 0
            for i in range(5):
                for j in range(5):
                    centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                    idx += 1
            centers[25] = [0.5, 0.5]
        else:
            # Quasi-random spread
            centers = np.random.uniform(0.1, 0.9, (N, 2))
            
        # Add controlled noise to break symmetry
        centers += np.random.uniform(-0.03, 0.03, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        radii = get_safe_radii(centers)
        v0 = np.zeros(3*N)
        v0[0::3] = centers[:, 0]
        v0[1::3] = centers[:, 1]
        v0[2::3] = radii
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=_BOUNDS, constraints=cons,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
            # Accept if sufficiently feasible
            if np.min(constraint_func(res.x)) >= -1e-6:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass

    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        for _ in range(30):
            v_pert = best_v.copy()
            # Slightly displace centers
            v_pert[0::3] += np.random.uniform(-0.008, 0.008, N)
            v_pert[1::3] += np.random.uniform(-0.008, 0.008, N)
            v_pert[0::3] = np.clip(v_pert[0::3], 0.01, 0.99)
            v_pert[1::3] = np.clip(v_pert[1::3], 0.01, 0.99)
            
            # Recompute safe radii for the perturbed geometry
            c_pert = np.column_stack((v_pert[0::3], v_pert[1::3]))
            v_pert[2::3] = get_safe_radii(c_pert)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=_BOUNDS, constraints=cons,
                               options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if np.min(constraint_func(res.x)) >= -1e-6:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # Phase 3: High-precision final polish on the absolute best configuration
    if best_v is not None:
        try:
            res_final = minimize(objective, best_v, method='SLSQP', bounds=_BOUNDS, constraints=cons,
                                 options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraint_func(res_final.x)) >= -1e-8:
                best_v = res_final.x
                best_sum = -res_final.fun
        except Exception:
            pass
            
    # Fallback safety net
    if best_v is None:
        centers = np.random.uniform(0.2, 0.8, (N, 2))
        radii = np.full(N, 0.05)
        best_v = np.zeros(3*N)
        best_v[0::3] = centers[:,0]
        best_v[1::3] = centers[:,1]
        best_v[2::3] = radii
        best_sum = np.sum(radii)
        
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = np.maximum(best_v[2::3], 0.0)
    return centers, radii, float(np.sum(radii))
