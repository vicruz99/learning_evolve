# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000018 (state cd1c4815) state=93f7f230 sum of radii=2.629868 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def unpack(vars):
    """Converts flat optimization vector to centers and radii arrays."""
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(N):
        centers[i, 0] = vars[3*i]
        centers[i, 1] = vars[3*i+1]
        radii[i] = vars[3*i+2]
    return centers, radii

def objective(vars):
    """Objective to minimize: negative sum of radii."""
    radii = unpack(vars)[1]
    return -np.sum(radii)

def constraints_func(vars):
    """Returns array of constraint values g(x) >= 0."""
    centers, radii = unpack(vars)
    cons = []
    # Boundary constraints
    for i in range(N):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        cons.append(x - r)
        cons.append(1.0 - x - r)
        cons.append(y - r)
        cons.append(1.0 - y - r)
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - radii[i] - radii[j])
    return np.array(cons)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def create_hex_init(rows_counts, r_init=0.085):
    """Generates a hexagonal lattice initialization."""
    centers = np.zeros((N, 2))
    radii = np.full(N, r_init)
    idx = 0
    y_step = np.sqrt(3) * r_init
    y = r_init
    for i, count in enumerate(rows_counts):
        x_start = r_init + (r_init if i % 2 == 1 else 0)
        for k in range(count):
            if idx < N:
                x = x_start + k * 2 * r_init
                centers[idx] = [x, y]
                idx += 1
        y += y_step
    # Fill remaining if needed
    while idx < N:
        centers[idx] = np.random.uniform(r_init, 1.0 - r_init, 2)
        idx += 1
    return centers, radii

def rotate_grid(centers, angle):
    """Rotates a grid of centers around the square center."""
    cx, cy = 0.5, 0.5
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    new_centers = np.zeros_like(centers)
    for i in range(N):
        dx = centers[i, 0] - cx
        dy = centers[i, 1] - cy
        new_centers[i, 0] = dx * cos_a - dy * sin_a + cx
        new_centers[i, 1] = dx * sin_a + dy * cos_a + cy
    return new_centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    configs = []
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 6, 6, 5, 4],
        [4, 6, 6, 6, 4],
        [5, 5, 5, 5, 6],
        [6, 6, 6, 4, 4]
    ]
    
    # Structured hexagonal starts
    for pat in patterns:
        c, r = create_hex_init(pat, r_init=0.085)
        configs.append((c, r))
        # Add rotated variants to break symmetry
        for angle in [0.15, 0.3, -0.15]:
            c_rot = rotate_grid(c, angle)
            configs.append((c_rot, r.copy()))
            
    # Random starts
    np.random.seed(42)
    for _ in range(10):
        c = np.random.uniform(0.15, 0.85, (N, 2))
        r = np.full(N, 0.06)
        configs.append((c, r))
        
    # Grid start
    c = np.zeros((N, 2))
    r = np.full(N, 0.09)
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < N:
                c[idx] = [0.1 + 0.2*j, 0.1 + 0.18*i]
                idx += 1
    configs.append((c, r))
    
    # Optimization loop
    for c_init, r_init in configs:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        # Perturb to escape exact symmetry
        x0 += np.random.normal(0, 1e-4, x0.shape)
        x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
        x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
        x0[2::3] = np.clip(x0[2::3], 0.01, 0.2)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            c_opt, r_opt = unpack(res.x)
            s = np.sum(r_opt)
            
            if s > best_sum:
                # Quick feasibility check
                valid = True
                for i in range(N):
                    if r_opt[i] < 0 or c_opt[i,0] < r_opt[i] - 1e-9 or c_opt[i,0] + r_opt[i] > 1 + 1e-9 or \
                       c_opt[i,1] < r_opt[i] - 1e-9 or c_opt[i,1] + r_opt[i] > 1 + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.sqrt(np.sum((c_opt[i] - c_opt[j]) ** 2))
                            if d < r_opt[i] + r_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                if valid and s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Final local refinement on best solution
    if best_centers is not None:
        x0 = np.zeros(3 * N)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = best_radii
        for _ in range(4):
            x0 += np.random.normal(0, 1e-5, x0.shape)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-14})
                c_opt, r_opt = unpack(res.x)
                s = np.sum(r_opt)
                if s > best_sum:
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
                    best_sum = s
            except Exception:
                break
                
    # Fallback safety
    if best_centers is None:
        best_centers, best_radii = create_hex_init([5, 6, 5, 6, 4], r_init=0.08)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
