# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 9a6065a6) state=ea64aff4 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars, n):
    """Compute all inequality constraints g(vars) >= 0."""
    xi = vars[0::3]
    yi = vars[1::3]
    ri = vars[2::3]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    res = np.concatenate([
        xi - ri,
        1.0 - xi - ri,
        yi - ri,
        1.0 - yi - ri
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xi[:, None] - xi[None, :]
    dy = yi[:, None] - yi[None, :]
    dr = ri[:, None] + ri[None, :]
    
    i_idx, j_idx = np.tril_indices(n, -1)
    overlap_cons = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2 - dr[i_idx, j_idx]**2
    
    return np.concatenate([res, overlap_cons])

def force_init(n, seed, steps=3000):
    """Generate a well-spread initial configuration using repulsive forces."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (n, 2))
    dt = 0.005
    
    for _ in range(steps):
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)  # Avoid self-interaction
        
        mask = dists < 0.35
        # Repulsive force proportional to 1/dist^3
        force_mag = np.where(mask, 1.0 / (dists**3), 0.0)
        forces = np.sum(force_mag[:, :, None] * diff, axis=1)
        
        # Wall repulsion to keep circles inside [0.05, 0.95]
        forces[:, 0] += np.where(centers[:, 0] < 0.15, 3.0, np.where(centers[:, 0] > 0.85, -3.0, 0.0))
        forces[:, 1] += np.where(centers[:, 1] < 0.15, 3.0, np.where(centers[:, 1] > 0.85, -3.0, 0.0))
        
        centers += dt * forces
        centers = np.clip(centers, 0.02, 0.98)
        
    # Compute safe initial radii based on nearest neighbors and walls
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
        for j in range(n):
            if i != j:
                d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0 * 0.92
    return centers, radii

def hex_init(n, seed, r=0.092):
    """Generate a perturbed hexagonal lattice initialization."""
    np.random.seed(seed)
    centers = []
    y = r
    row = 0
    while len(centers) < n and y + r <= 1.0:
        x = r + (row % 2) * r
        while len(centers) < n and x + r <= 1.0:
            centers.append([x, y])
            x += 2 * r
        y += np.sqrt(3) * r
        row += 1
        
    while len(centers) < n:
        centers.append([np.random.uniform(r, 1 - r), np.random.uniform(r, 1 - r)])
        
    centers = np.array(centers[:n])
    centers += np.random.normal(0, 0.008, centers.shape)
    centers = np.clip(centers, r, 1 - r)
    return centers, np.full(n, r)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_vars = None
    
    # Generate diverse initial configurations
    inits = []
    for s in range(15):
        inits.append(force_init(n, s))
    for s in range(10):
        inits.append(hex_init(n, s))
        
    # Phase 1: Multi-start optimization
    for c_init, r_init in inits:
        x0 = np.zeros(3 * n)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            s_val = -res.fun
            if s_val > best_sum:
                c_vals = constraints(res.x, n)
                if np.min(c_vals) >= -1e-6:
                    best_sum = s_val
                    best_vars = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement to escape shallow local minima
    if best_vars is not None:
        for _ in range(5):
            x_pert = best_vars + np.random.normal(0, 0.0008, 3 * n)
            for i in range(n):
                r = max(0.01, x_pert[3 * i + 2])
                x_pert[3 * i + 2] = r
                x_pert[3 * i] = np.clip(x_pert[3 * i], r, 1.0 - r)
                x_pert[3 * i + 1] = np.clip(x_pert[3 * i + 1], r, 1.0 - r)
                
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                s_val = -res.fun
                c_vals = constraints(res.x, n)
                if np.min(c_vals) >= -1e-6 and s_val > best_sum:
                    best_sum = s_val
                    best_vars = res.x.copy()
            except Exception:
                pass

    # Fallback configuration
    if best_vars is None:
        best_vars = np.zeros(3 * n)
        best_vars[2::3] = 0.09
        best_vars[0::3] = np.repeat(np.linspace(0.1, 0.9, 5), 6)[:n]
        best_vars[1::3] = np.tile(np.linspace(0.1, 0.9, 6), 5)[:n]
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Phase 3: Strict validity check and minimal repair
    for _ in range(100):
        valid = True
        for i in range(n):
            if radii[i] < 0:
                valid = False
                break
            if centers[i,0] - radii[i] < -1e-9 or centers[i,0] + radii[i] > 1.0 + 1e-9:
                valid = False
                break
            if centers[i,1] - radii[i] < -1e-9 or centers[i,1] + radii[i] > 1.0 + 1e-9:
                valid = False
                break
                
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1]) < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            break
        radii *= 0.995
        
    return centers, radii, float(np.sum(radii))
