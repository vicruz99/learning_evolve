# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000034 (state 93f7f230) state=0d4d18bd sum of radii=2.627928 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
TRI_INDICES = np.triu_indices(N_CIRCLES, 1)

def unpack(vars):
    """Converts flat optimization vector to centers and radii arrays."""
    reshaped = vars.reshape(N_CIRCLES, 3)
    centers = reshaped[:, :2].copy()
    radii = reshaped[:, 2].copy()
    return centers, radii

def objective(vars):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Returns vector of inequality constraints g(vars) >= 0."""
    centers = vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = vars.reshape(N_CIRCLES, 3)[:, 2]
    x, y = centers[:, 0], centers[:, 1]
    
    n_cons = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    cons = np.empty(n_cons)
    
    # Boundary constraints
    cons[:N_CIRCLES] = x - radii
    cons[N_CIRCLES:2*N_CIRCLES] = 1.0 - x - radii
    cons[2*N_CIRCLES:3*N_CIRCLES] = y - radii
    cons[3*N_CIRCLES:4*N_CIRCLES] = 1.0 - y - radii
    
    # Overlap constraints: dist >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = radii[:, None] + radii[None, :]
    
    cons[4*N_CIRCLES:] = dist[TRI_INDICES] - r_sum[TRI_INDICES]
    return cons

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return bounds

def generate_hex_init(counts, r_start=0.085):
    """Generates a hexagonal lattice initialization."""
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, r_start)
    idx = 0
    y = r_start
    y_step = np.sqrt(3) * r_start
    for i, cnt in enumerate(counts):
        x_start = r_start + (r_start if i % 2 == 1 else 0.0)
        for k in range(cnt):
            if idx < N_CIRCLES:
                x = x_start + k * 2 * r_start
                centers[idx] = [x, y]
                idx += 1
        y += y_step
    while idx < N_CIRCLES:
        centers[idx] = np.random.uniform(r_start, 1.0 - r_start, 2)
        idx += 1
    return centers, radii

def rotate_points(centers, angle):
    """Rotates a grid of centers around the square center."""
    cx, cy = 0.5, 0.5
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = centers[:, 0] - cx
    dy = centers[:, 1] - cy
    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy
    return np.column_stack([new_x, new_y])

def is_strictly_valid(centers, radii, tol=1e-9):
    """Checks feasibility with a given tolerance."""
    n = centers.shape[0]
    if np.any(radii < 0):
        return False
    if np.any(centers[:, 0] - radii < -tol) or np.any(centers[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(centers[:, 1] - radii < -tol) or np.any(centers[:, 1] + radii > 1.0 + tol):
        return False
    
    c1 = centers[:, None, :]
    c2 = centers[None, :, :]
    dists = np.sqrt(np.sum((c1 - c2)**2, axis=2))
    r_sum = radii[:, None] + radii[None, :]
    return np.all(dists[TRI_INDICES] >= r_sum[TRI_INDICES] - tol)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    best_sum = -1.0
    best_vars = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 6, 6, 5, 4],
        [4, 6, 6, 6, 4],
        [5, 5, 5, 5, 6],
        [6, 6, 6, 4, 4],
        [5, 7, 5, 6, 3],
        [6, 6, 5, 5, 4],
        [4, 5, 6, 5, 6]
    ]
    
    inits = []
    for pat in patterns:
        c, r = generate_hex_init(pat, r_start=0.085)
        inits.append((c, r))
        for ang in [0.15, -0.15, 0.3, -0.3, 0.45]:
            inits.append((rotate_points(c, ang), r.copy()))
            
    np.random.seed(42)
    for _ in range(20):
        c = np.random.uniform(0.15, 0.85, (N_CIRCLES, 2))
        r = np.full(N_CIRCLES, 0.06)
        inits.append((c, r))
        
    # Grid initialization
    c = np.zeros((N_CIRCLES, 2))
    r = np.full(N_CIRCLES, 0.09)
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < N_CIRCLES:
                c[idx] = [0.1 + 0.18*j, 0.1 + 0.16*i]
                idx += 1
    inits.append((c, r))
    
    # Primary optimization phase
    for c_init, r_init in inits:
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        # Ensure strict feasibility for robust SLSQP start
        x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
        x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
        x0[2::3] = np.clip(x0[2::3], 0.01, 0.2)
        
        # Perturb to break exact symmetries
        x0 += np.random.normal(0, 1e-5, x0.shape)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    c_opt, r_opt = unpack(res.x)
                    if is_strictly_valid(c_opt, r_opt):
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Iterative refinement phase to escape local minima
    if best_vars is not None:
        for step in range(10):
            noise_scale = 0.003 / (step + 1)
            x_pert = best_vars + np.random.normal(0, noise_scale, best_vars.shape)
            
            # Project perturbed variables back to feasible bounds
            r_pert = np.maximum(x_pert[2::3], 0.005)
            x_pert[0::3] = np.clip(x_pert[0::3], r_pert, 1.0 - r_pert)
            x_pert[1::3] = np.clip(x_pert[1::3], r_pert, 1.0 - r_pert)
            x_pert[2::3] = r_pert
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        c_opt, r_opt = unpack(res.x)
                        if is_strictly_valid(c_opt, r_opt):
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization completely fails
    if best_vars is None:
        best_vars = np.zeros(3 * N_CIRCLES)
        best_vars[2::3] = 0.06
        best_vars[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N_CIRCLES]
        best_vars[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N_CIRCLES]
        
    centers = best_vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = best_vars.reshape(N_CIRCLES, 3)[:, 2]
    
    # Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        if is_strictly_valid(centers, radii, tol=1e-12):
            break
        radii *= 0.9999
        
    return centers, radii, float(np.sum(radii))
