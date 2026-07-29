# sol_000046 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 9a6065a6) state=e13e60e1 sum of radii=2.619475 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Compute all inequality constraints g(x) >= 0.
    Returns a 1D array containing boundary and non-overlap constraints.
    """
    n = len(x) // 3
    xc = x[0::3]
    yc = x[1::3]
    rc = x[2::3]
    
    num_overlaps = n * (n - 1) // 2
    c = np.empty(n * 4 + num_overlaps)
    
    # Boundary constraints
    c[:n] = xc - rc
    c[n:2*n] = 1.0 - xc - rc
    c[2*n:3*n] = yc - rc
    c[3*n:4*n] = 1.0 - yc - rc
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    dr = rc[:, None] + rc[None, :]
    
    i_idx, j_idx = np.triu_indices(n, 1)
    c[4*n:] = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2 - dr[i_idx, j_idx]**2
    
    return c

def force_directed_init(n, seed):
    """Generate initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    centers = np.random.uniform(0.2, 0.8, (n, 2))
    
    for _ in range(1500):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                d2 = dx * dx + dy * dy
                if d2 < 0.04 and d2 > 1e-6:
                    f = 0.008 / d2
                    fx, fy = f * dx, f * dy
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
        centers += forces * 0.01
        centers = np.clip(centers, 0.05, 0.95)
        
    radii = np.full(n, 0.075)
    return centers, radii

def hex_grid_init(n, seed):
    """Generate a perturbed hexagonal lattice initialization."""
    np.random.seed(seed)
    r_init = 0.085 + np.random.uniform(-0.005, 0.005)
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    
    idx = 0
    y = r_init
    row = 0
    while idx < n and y + r_init <= 1.0:
        x = r_init + (row % 2) * r_init
        while idx < n and x + r_init <= 1.0:
            centers[idx] = [x, y]
            x += 2 * r_init
            idx += 1
        y += r_init * np.sqrt(3.0)
        row += 1
        
    while idx < n:
        centers[idx] = np.random.uniform(r_init, 1.0 - r_init, 2)
        idx += 1
        
    # Add controlled noise to break symmetry
    centers += np.random.normal(0, 0.002, centers.shape)
    return centers, radii

def project_to_feasible(centers, radii, n):
    """Project centers to strictly satisfy boundary constraints given radii."""
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
    return centers

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -np.inf
    best_x = None
    
    # Phase 1: Multiple diverse starts
    # Combine force-directed and hexagonal initializations
    inits = []
    for s in range(8):
        inits.append(('force', s))
    for s in range(7):
        inits.append(('hex', s))
        
    for itype, seed in inits:
        if itype == 'force':
            c_init, r_init = force_directed_init(n, seed)
        else:
            c_init, r_init = hex_grid_init(n, seed)
            
        # Ensure strict initial feasibility for SLSQP
        c_init = project_to_feasible(c_init, r_init, n)
        
        x0 = np.zeros(3 * n)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            # Check feasibility and improvement
            c_vals = constraints(res.x)
            if np.min(c_vals) >= -1e-9 and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Phase 2: Local refinement to escape shallow minima
    if best_x is not None:
        for _ in range(6):
            # Perturb positions and radii slightly
            noisy = best_x + np.random.normal(0, 1.5e-4, size=best_x.shape)
            
            # Project radii to positive and clip centers
            for i in range(n):
                r = max(1e-6, noisy[3*i + 2])
                noisy[3*i + 2] = r
                noisy[3*i] = np.clip(noisy[3*i], r, 1.0 - r)
                noisy[3*i + 1] = np.clip(noisy[3*i + 1], r, 1.0 - r)
                
            try:
                res = minimize(objective, noisy, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-9 and -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x.copy()
            except Exception:
                break

    # Fallback (should rarely trigger)
    if best_x is None:
        best_x = np.zeros(3 * n)
        best_x[2::3] = 0.08
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:n]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:n]
        best_val = np.sum(best_x[2::3])

    # Extract results
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Safety repair loop to guarantee strict validity against numerical drift
    for _ in range(50):
        valid = True
        for i in range(n):
            if radii[i] < 0:
                valid = False
                break
            if centers[i, 0] - radii[i] < -1e-10 or centers[i, 0] + radii[i] > 1.0 + 1e-10:
                valid = False
                break
            if centers[i, 1] - radii[i] < -1e-10 or centers[i, 1] + radii[i] > 1.0 + 1e-10:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if dist < radii[i] + radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
        # Minimal shrinkage to recover validity
        radii *= 0.998
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
