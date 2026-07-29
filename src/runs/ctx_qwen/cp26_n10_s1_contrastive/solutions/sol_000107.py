# sol_000107 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000080 (state b3333e60) state=4b3ba205 sum of radii=2.623489 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
_BOUNDS = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
_MASK = np.triu(np.ones((N, N), dtype=bool), k=1)

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def constraint_func(v):
    """
    Computes inequality constraints g(v) >= 0.
    Includes boundary containment and pairwise non-overlap.
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
    
    c.append(dist2[_MASK] - rs[_MASK]**2)
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
        # 0.95 factor ensures strict feasibility for SLSQP start
        r[i] = 0.95 * min(d_wall, 0.5 * d_min)
    return np.maximum(r, 1e-6)

def generate_hex_init(rotation=0.0, scale=1.0, jitter=0.015):
    """Generate a hexagonal lattice initialization with optional rotation and jitter."""
    rng = np.random.default_rng(42)
    pts = []
    r_est = 0.095 * scale
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
        
    pts = np.array(pts[:N])
    
    # Center and scale
    pts = pts - 0.5
    pts = pts * scale
    pts = pts + 0.5
    
    # Apply rotation
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = pts @ rot_mat.T
        pts = pts - pts.mean(axis=0) + 0.5
        
    # Add controlled jitter
    jitter_mat = rng.uniform(-jitter, jitter, pts.shape)
    pts = pts + jitter_mat
    pts = np.clip(pts, 0.02, 0.98)
    return pts

def generate_grid_init(jitter=0.02):
    """Generate a 5x5 grid + center initialization with jitter."""
    rng = np.random.default_rng(77)
    pts = np.zeros((N, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            pts[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
            idx += 1
    pts[25] = [0.5, 0.5]
    
    pts += rng.uniform(-jitter, jitter, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    best_sum = -np.inf
    best_v = None
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Phase 1: Diverse restarts to explore global landscape
    # We try multiple rotations and scales of hex lattice, plus grid inits
    configs = []
    for rot in np.linspace(-0.15, 0.15, 15):
        configs.append(('hex', rot, 1.0))
    for rot in [0.0, 0.05, -0.05]:
        configs.append(('hex', rot, 0.95))
        configs.append(('hex', rot, 1.05))
    for _ in range(20):
        configs.append(('grid', 0.0, 0.0))

    for i, cfg in enumerate(configs):
        if cfg[0] == 'hex':
            centers_init = generate_hex_init(rotation=cfg[1], scale=cfg[2], jitter=0.012)
        else:
            centers_init = generate_grid_init(jitter=0.02)
            
        radii_init = get_safe_radii(centers_init)
        v0 = np.zeros(3*N)
        v0[0::3] = centers_init[:, 0]
        v0[1::3] = centers_init[:, 1]
        v0[2::3] = radii_init
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=_BOUNDS, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            
            cons_val = constraint_func(res.x)
            if np.min(cons_val) >= -1e-7:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue

    # Phase 2: Basin-hopping / Local perturbation refinement
    if best_v is not None:
        rng = np.random.default_rng(123)
        for _ in range(60):
            v_pert = best_v.copy()
            
            # Perturb centers significantly enough to change contact graph
            v_pert[0::3] += rng.uniform(-0.005, 0.005, N)
            v_pert[1::3] += rng.uniform(-0.005, 0.005, N)
            v_pert[0::3] = np.clip(v_pert[0::3], 0.01, 0.99)
            v_pert[1::3] = np.clip(v_pert[1::3], 0.01, 0.99)
            
            # Recompute safe radii for new geometry
            c_pert = np.column_stack((v_pert[0::3], v_pert[1::3]))
            v_pert[2::3] = get_safe_radii(c_pert)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=_BOUNDS, constraints=cons,
                               options={'maxiter': 2500, 'ftol': 1e-13, 'disp': False})
                
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                continue

    # Phase 3: High-precision final polish on the absolute best configuration
    if best_v is not None:
        try:
            res_final = minimize(objective, best_v, method='SLSQP', bounds=_BOUNDS, constraints=cons,
                                 options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraint_func(res_final.x)) >= -1e-8:
                best_v = res_final.x
                best_sum = -res_final.fun
        except Exception:
            pass
            
    # Fallback safety net (should not be reached with robust inits)
    if best_v is None:
        centers = generate_grid_init(jitter=0.0)
        radii = get_safe_radii(centers)
        best_v = np.zeros(3*N)
        best_v[0::3] = centers[:,0]
        best_v[1::3] = centers[:,1]
        best_v[2::3] = radii
        best_sum = np.sum(radii)
        
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = np.maximum(best_v[2::3], 0.0)
    return centers, radii, float(np.sum(radii))
