# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000052 (state 0d4d18bd) state=63a33892 sum of radii=2.630172 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
TRIL_IDX = np.tril_indices(N_CIRCLES, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Overlap constraints using squared distances for better gradient behavior
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    b = []
    for _ in range(N_CIRCLES):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def init_hex(row_counts, r0):
    """Generates a hexagonal lattice initialization."""
    centers = np.zeros((N_CIRCLES, 2))
    idx = 0
    y = r0
    y_step = np.sqrt(3.0) * r0
    for i, cnt in enumerate(row_counts):
        x_start = r0 + (r0 if i % 2 == 1 else 0.0)
        for k in range(cnt):
            if idx < N_CIRCLES:
                centers[idx, 0] = x_start + k * 2.0 * r0
                centers[idx, 1] = y
                idx += 1
        y += y_step
    # Fill remaining if any
    while idx < N_CIRCLES:
        centers[idx, 0] = np.random.uniform(0.1, 0.9)
        centers[idx, 1] = np.random.uniform(0.1, 0.9)
        idx += 1
    return centers

def init_force(seed):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (N_CIRCLES, 2))
    for _ in range(600):
        forces = np.zeros_like(centers)
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.hypot(dx, dy)
                if dist < 0.28 and dist > 1e-5:
                    f = 0.012 / (dist**2 + 0.001)
                    fx, fy = f * dx, f * dy
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
        # Repel from edges to utilize interior space
        for i in range(N_CIRCLES):
            if centers[i, 0] < 0.12: forces[i, 0] += 0.025
            elif centers[i, 0] > 0.88: forces[i, 0] -= 0.025
            if centers[i, 1] < 0.12: forces[i, 1] += 0.025
            elif centers[i, 1] > 0.88: forces[i, 1] -= 0.025
        centers += forces * 0.06
        centers = np.clip(centers, 0.05, 0.95)
    return centers

def make_x0(centers, r_val):
    """Flatten centers and radii into optimization vector."""
    x0 = np.zeros(3 * N_CIRCLES)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = r_val
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    best_sum = -1.0
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    configs = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 6, 6, 5, 4],
        [4, 6, 6, 6, 4], [5, 5, 5, 5, 6], [6, 6, 6, 4, 4],
        [5, 7, 5, 6, 3], [6, 6, 5, 5, 4], [4, 5, 6, 5, 6]
    ]
    
    # Hexagonal lattices with rotations
    for pat in patterns:
        c = init_hex(pat, 0.083)
        configs.append((c, 0.083))
        for ang in [0.1, -0.1, 0.2, -0.2, 0.35, 0.5, -0.5]:
            cx, cy = 0.5, 0.5
            ca, sa = np.cos(ang), np.sin(ang)
            dx = c[:, 0] - cx
            dy = c[:, 1] - cy
            c_rot = np.column_stack([dx*ca - dy*sa + cx, dx*sa + dy*ca + cy])
            configs.append((c_rot, 0.083))
            
    # Force-directed starts
    for s in range(8):
        configs.append((init_force(s), 0.055))
        
    # Grid start
    c_grid = np.zeros((N_CIRCLES, 2))
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < N_CIRCLES:
                c_grid[idx] = [0.1 + 0.18*j, 0.1 + 0.16*i]
                idx += 1
    configs.append((c_grid, 0.075))
    
    # Phase 1: Multi-start optimization
    for c_init, r_init in configs:
        x0 = make_x0(c_init, r_init)
        x0 += np.random.normal(0, 1e-5, x0.shape)
        x0[0::3] = np.clip(x0[0::3], r_init, 1.0 - r_init)
        x0[1::3] = np.clip(x0[1::3], r_init, 1.0 - r_init)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative deflation & refinement to escape local minima
    if best_x is not None:
        for step in range(15):
            noise = 0.0025 / (step + 1)
            x_pert = best_x + np.random.normal(0, noise, best_x.shape)
            # Shrink radii slightly to guarantee feasibility for optimizer
            r_pert = np.maximum(x_pert[2::3] * 0.982, 0.005)
            x_pert[2::3] = r_pert
            x_pert[0::3] = np.clip(x_pert[0::3], r_pert, 1.0 - r_pert)
            x_pert[1::3] = np.clip(x_pert[1::3], r_pert, 1.0 - r_pert)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun):
                    curr_sum = -res.fun
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-7 and curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Fallback
    if best_x is None:
        best_x = make_x0(init_hex([5,6,5,6,4], 0.08), 0.08)
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N_CIRCLES):
            if radii[i] < 0: valid=False; break
            if centers[i,0] - radii[i] < -1e-12 or centers[i,0] + radii[i] > 1.0 + 1e-12: valid=False; break
            if centers[i,1] - radii[i] < -1e-12 or centers[i,1] + radii[i] > 1.0 + 1e-12: valid=False; break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if d < radii[i] + radii[j] - 1e-12:
                        valid=False; break
                if not valid: break
        if valid: break
        radii *= 0.9999
        
    return centers, radii, float(np.sum(radii))
