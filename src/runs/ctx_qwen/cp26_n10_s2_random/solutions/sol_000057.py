# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000036 (state ae916370) state=9e7ec844 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(p):
    """Computes all boundary and non-overlap constraint values (must be >= 0)."""
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_boundary = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-16)  # epsilon prevents domain errors
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_overlap = dist[mask] - dr[mask]
    
    return np.concatenate([c_boundary, c_overlap])

def objective(p):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(p[2::3])

def get_bounds():
    """Defines variable bounds: x,y in [0,1], r in [1e-6, 0.5]."""
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return bounds

def generate_hex_init(seed):
    """Generates a hexagonal lattice initialization with small noise."""
    rng = np.random.default_rng(seed)
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    while len(pts) < N:
        x_start = r_est if row % 2 == 0 else 2 * r_est
        x = x_start
        while x + r_est <= 1.0 and len(pts) < N:
            pts.append([x, y])
            x += 2 * r_est
        y += r_est * np.sqrt(3)
        row += 1
    # Fill remaining if any
    while len(pts) < N:
        pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
        
    pts = np.array(pts[:N]) + rng.normal(0, 0.003, (N, 2))
    pts = np.clip(pts, 0.05, 0.95)
    
    p0 = np.zeros(3 * N)
    p0[0::3] = pts[:, 0]
    p0[1::3] = pts[:, 1]
    p0[2::3] = r_est
    return p0

def generate_grid_init(seed):
    """Generates a structured grid initialization with small noise."""
    rng = np.random.default_rng(seed)
    gs = np.linspace(0.12, 0.88, 6)
    pts = []
    for x in gs:
        for y in gs:
            if len(pts) < N:
                pts.append([x, y])
    while len(pts) < N:
        pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
        
    pts = np.array(pts[:N]) + rng.normal(0, 0.004, (N, 2))
    pts = np.clip(pts, 0.05, 0.95)
    
    p0 = np.zeros(3 * N)
    p0[0::3] = pts[:, 0]
    p0[1::3] = pts[:, 1]
    p0[2::3] = 0.075
    return p0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_p = None
    best_val = -np.inf
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Phase 1: Diverse initializations to locate promising basins
    for seed in range(8):
        for init_fn in [generate_hex_init, generate_grid_init]:
            x0 = init_fn(seed)
            try:
                res = minimize(
                    objective, x0, method='SLSQP', bounds=bounds,
                    constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False}
                )
                c_check = compute_constraints(res.x)
                if np.min(c_check) >= -1e-7:
                    curr_val = -res.fun
                    if curr_val > best_val:
                        best_val = curr_val
                        best_p = res.x.copy()
            except Exception:
                continue
                
    # Phase 2: Perturbation-based local search to escape local minima
    if best_p is not None:
        rng = np.random.default_rng(123)
        for i in range(15):
            p_pert = best_p.copy()
            # Gradually decreasing perturbation scale
            scale = 0.0015 * (1.0 - i / 20.0)
            p_pert += rng.normal(0, scale, p_pert.shape)
            p_pert = np.clip(p_pert, 0.0, 1.0)
            p_pert[2::3] = np.clip(p_pert[2::3], 1e-6, 0.5)
            
            try:
                res = minimize(
                    objective, p_pert, method='SLSQP', bounds=bounds,
                    constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False}
                )
                c_check = compute_constraints(res.x)
                if np.min(c_check) >= -1e-7:
                    curr_val = -res.fun
                    if curr_val > best_val:
                        best_val = curr_val
                        best_p = res.x.copy()
            except Exception:
                continue
                
    # Fallback if optimization unexpectedly fails
    if best_p is None:
        best_p = generate_hex_init(0)
        
    # Final safety repair to guarantee strict feasibility within validator tolerance
    for _ in range(50):
        c_vals = compute_constraints(best_p)
        min_c = np.min(c_vals)
        if min_c < -1e-9:
            # Shrink radii proportionally to resolve overlaps/boundary violations
            best_p[2::3] *= 0.999
        else:
            break
            
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3].copy()
    radii = np.maximum(radii, 0.0)
    
    # Strict boundary clamping as final safeguard
    for i in range(N):
        max_allowed = min(centers[i, 0], 1.0 - centers[i, 0], 
                          centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > max_allowed:
            radii[i] = max_allowed
            
    return centers, radii, float(np.sum(radii))
