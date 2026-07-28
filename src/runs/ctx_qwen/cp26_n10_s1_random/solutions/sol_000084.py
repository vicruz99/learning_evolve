# sol_000084 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000032 (state ac51bd1a) state=1af69af5 sum of radii=2.333622 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars[2*N:])

def get_constraints(vars):
    """Compute inequality constraints >= 0 for valid packing"""
    cx = vars[:N]
    cy = vars[N:2*N]
    r = vars[2*N:]
    
    con = []
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    con.append(cx - r)
    con.append(1.0 - cx - r)
    con.append(cy - r)
    con.append(1.0 - cy - r)
    
    # Overlap constraints: squared distance >= squared sum of radii
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx*dx + dy*dy
    r_sum = r[:, None] + r[None, :]
    
    # Only upper triangle to avoid duplicates
    i, j = np.triu_indices(N, k=1)
    con.append(dist_sq[i, j] - r_sum[i, j]**2)
    
    return np.concatenate(con)

def generate_hex_config(row_counts, scale=0.92):
    """Generate a hexagonal lattice configuration for N circles"""
    pts = []
    r0 = 0.1
    dy = np.sqrt(3) * r0
    y = r0
    for idx, cnt in enumerate(row_counts):
        shift = r0 if idx % 2 == 1 else 0.0
        row_width = (cnt - 1) * 2 * r0
        x_start = (1.0 - row_width) / 2.0 + shift
        for _ in range(cnt):
            pts.append([x_start, y])
            x_start += 2 * r0
        y += dy
        
    pts = np.array(pts[:N])
    # Normalize to fit within [0, 1] with padding
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0)) * scale + (1.0 - scale) / 2.0
    return np.clip(pts, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Diverse row distributions for 26 circles that approximate hexagonal packing
    row_dists = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 6, 4], [4, 6, 6, 6, 4], [5, 5, 5, 6, 5],
        [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [7, 5, 6, 4, 4]
    ]
    
    np.random.seed(42)
    configs = []
    
    # Generate structured hexagonal starts
    for rc in row_dists:
        if sum(rc) >= N:
            configs.append(generate_hex_config(rc, 0.92))
            configs.append(generate_hex_config(rc, 0.95))
            
    # Add perturbed versions to escape symmetry traps
    for _ in range(12):
        base = configs[np.random.randint(len(configs))]
        cfg = base + np.random.uniform(-0.025, 0.025, (N, 2))
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # Add a regular grid + center start
    grid = np.array([(i*0.2+0.1, j*0.2+0.1) for j in range(5) for i in range(5)])
    grid = np.vstack([grid, [0.5, 0.5]])
    configs.append(grid)
    
    # Optimization loop
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), np.full(N, 0.09)])
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-9, 'disp': False})
            
            if np.isfinite(res.fun):
                cx = res.x[:N]
                cy = res.x[N:2*N]
                centers = np.column_stack((cx, cy))
                
                # Strict post-processing: compute exact max feasible radii for these centers
                # This guarantees validity and often recovers margin lost to numerical tolerance
                radii_adj = np.zeros(N)
                for i in range(N):
                    d_wall = min(cx[i], 1.0-cx[i], cy[i], 1.0-cy[i])
                    
                    # Pairwise distances
                    dists = np.sqrt((cx-cx[i])**2 + (cy-cy[i])**2)
                    dists[i] = np.inf
                    d_pair = np.min(dists) / 2.0
                    
                    # Apply tiny buffer to strictly satisfy checker tolerance
                    radii_adj[i] = min(d_wall, d_pair) * 0.9999999
                    
                s = np.sum(radii_adj)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers
                    best_radii = radii_adj
        except Exception:
            continue
            
    # Fallback to a known valid configuration if optimization fails unexpectedly
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(N, 0.08)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
