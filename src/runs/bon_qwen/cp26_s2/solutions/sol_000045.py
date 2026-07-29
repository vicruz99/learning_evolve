# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 256846a1) state=0f9fbdd4 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def get_constraints(z):
    """Returns inequality constraints g(z) >= 0 for SLSQP."""
    n = N_CIRCLES
    cons = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    for i in range(n):
        x, y, r = z[3*i], z[3*i+1], z[3*i+2]
        cons.append(x - r)
        cons.append(1.0 - x - r)
        cons.append(y - r)
        cons.append(1.0 - y - r)
    # Non-overlap constraints: dist^2 >= (r1 + r2)^2
    for i in range(n):
        xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
        for j in range(i+1, n):
            xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
            dx = xi - xj
            dy = yi - yj
            cons.append(dx*dx + dy*dy - (ri + rj)**2)
    return np.array(cons)

def obj_func(z):
    """Negative sum of radii for minimization."""
    return -np.sum(z[2::3])

def make_initial_hex(n):
    """Generates a hexagonal lattice initial configuration."""
    pts = []
    # Row configuration summing to 26
    row_counts = [6, 5, 6, 5, 4]
    s = 0.16
    h = s * np.sqrt(3) / 2
    y_pos = 0.15
    for i, count in enumerate(row_counts):
        width = (count - 1) * s
        x_start = 0.5 - width / 2 + (i % 2) * s / 2
        for k in range(count):
            x = x_start + k * s
            pts.append((x, y_pos))
        y_pos += h
        
    z0 = np.zeros(3 * n)
    for i, (x, y) in enumerate(pts):
        z0[3*i] = x
        z0[3*i+1] = y
        z0[3*i+2] = 0.08  # Initialize with reasonable radius
    return z0

def run_packing():
    np.random.seed(42)
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_z = None
    best_val = 0.0
    
    # Generate initial configurations
    starts = [make_initial_hex(n)]
    for seed in [10, 20, 30]:
        np.random.seed(seed)
        # Random centers in [0.2, 0.8], small radii to start
        centers_init = np.random.uniform(0.2, 0.8, 2*n)
        radii_init = np.full(n, 0.03)
        z_rand = np.concatenate([centers_init, radii_init])
        starts.append(z_rand)
        
    # Optimization loop
    for z0 in starts:
        try:
            res = minimize(
                obj_func,
                z0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_z = res.x
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_z is None:
        best_z = make_initial_hex(n)
        
    centers = best_z[:2*n].reshape((n, 2))
    radii = best_z[2::3].copy()
    
    # Strict constraint enforcement for numerical safety
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1.0 - x, y, 1.0 - y)
        
    # Iterative overlap resolution to guarantee validity
    for _ in range(10):
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-9:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    
    return centers, radii, float(np.sum(radii))
