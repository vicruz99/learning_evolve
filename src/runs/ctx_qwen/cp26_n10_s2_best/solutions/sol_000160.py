# sol_000160 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=194e78bf sum of radii=2.617186 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist

N = 26

def get_optimal_radii(centers):
    """
    Solves an LP to find radii that maximize sum(r_i) for fixed centers.
    Constraints: r_i + r_j <= dist(i,j) and r_i <= dist to boundaries.
    """
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, 1.0)  # Placeholder, diagonal not used in constraints
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Maximum radius allowed by boundaries
    b_x = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    b_y = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    b_max = np.minimum(b_x, b_y)
    
    # LP Formulation: minimize -sum(r) subject to A_ub * r <= b_ub
    num_pairs = len(d_ij)
    A_ub = np.zeros((num_pairs + N, N))
    b_ub = np.zeros(num_pairs + N)
    
    # Non-overlap constraints: r_i + r_j <= d_ij
    A_ub[:num_pairs, i_idx] = 1.0
    A_ub[:num_pairs, j_idx] = 1.0
    b_ub[:num_pairs] = d_ij
    
    # Boundary constraints: r_i <= b_max_i
    A_ub[num_pairs:, np.arange(N)] = 1.0
    b_ub[num_pairs:] = b_max
    
    # Add tiny slack to ensure strict feasibility for the subsequent SLSQP start
    b_ub *= 0.99995
    
    c_obj = -np.ones(N)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    
    if res.success:
        return res.x
    return np.full(N, 0.01)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v, i_idx, j_idx):
    """
    Inequality constraints: boundaries and non-overlap (squared distances).
    All elements must be >= 0.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        (x[i_idx] - x[j_idx])**2 + (y[i_idx] - y[j_idx])**2 - (r[i_idx] + r[j_idx])**2
    ])
    return c

def generate_initial_configs():
    """Generates a diverse set of initial center configurations."""
    configs = []
    
    # 1. Hexagonal lattices with varying densities and rotations
    for r0 in [0.085, 0.095, 0.105]:
        for rot in np.linspace(-0.15, 0.15, 5):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 10:
                x_start = r0 + (row % 2) * r0
                x = x_start
                while x <= 1.0 - r0 and len(pts) < N + 10:
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3.0) * r0
                row += 1
            pts = np.array(pts[:N+10])
            
            # Rotate around center
            cx, cy = 0.5, 0.5
            pts[:, 0] -= cx; pts[:, 1] -= cy
            c, s = np.cos(rot), np.sin(rot)
            pts = pts @ np.array([[c, -s], [s, c]])
            pts[:, 0] += cx; pts[:, 1] += cy
            
            configs.append(np.clip(pts[:N], 0.02, 0.98))
            
    # 2. Staggered grids
    for scale in [0.17, 0.19, 0.21]:
        pts = np.array([[i*scale + 0.05, j*scale + 0.05] for i in range(6) for j in range(5)])
        configs.append(np.clip(pts[:N], 0.02, 0.98))
        
    # 3. Random dense scatters
    np.random.seed(42)
    for seed in range(8):
        np.random.seed(seed)
        configs.append(np.random.uniform(0.1, 0.9, (N, 2)))
        
    return configs

def run_packing():
    i_idx, j_idx = np.triu_indices(N, k=1)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (i_idx, j_idx)}
    
    best_sum = -1.0
    best_v = None
    
    initial_configs = generate_initial_configs()
    
    # Phase 1: Multi-start optimization from diverse topological seeds
    for cfg in initial_configs:
        r_init = get_optimal_radii(cfg)
        v0 = np.concatenate([cfg[:, 0], cfg[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict, 
                           options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Verify feasibility with tolerance
                if np.all(constraints(res.x, i_idx, j_idx) >= -1e-7):
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Aggressive perturbation & LP-reinitialization loop to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(40):
            np.random.seed(step + 5000)
            pert = current_v.copy()
            
            # Perturb centers to explore new basin
            pert[:2*N] += np.random.uniform(-0.006, 0.006, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            
            # Shrink radii significantly to guarantee feasibility after center shift
            pert[2*N:] *= 0.90
            
            # Recompute optimal radii for the perturbed centers using LP
            r_opt = get_optimal_radii(pert[:2*N].reshape(N, 2))
            pert[2*N:] = r_opt
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
                
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    if np.all(constraints(res.x, i_idx, j_idx) >= -1e-7):
                        best_sum = curr_sum
                        best_v = res.x.copy()
                        current_v = best_v.copy()
            except Exception:
                continue
                
    # Fallback in case optimization fails
    if best_v is None:
        cfg = initial_configs[0]
        best_v = np.concatenate([cfg[:, 0], cfg[:, 1], get_optimal_radii(cfg)])
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validation compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], 
                       centers[i, 1], 1.0 - centers[i, 1])
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(30):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
