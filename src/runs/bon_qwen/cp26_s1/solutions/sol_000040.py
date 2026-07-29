# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6f5bcc91) state=86db94db sum of radii=1.040000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26


def objective(params):
    radii = params[2 * N:]
    return -np.sum(radii)


def constraint_function(params):
    centers = params[:2 * N].reshape(N, 2)
    radii = params[2 * N:]
    
    c = []
    
    # Boundary constraints: circles inside [0,1]x[0,1]
    for i in range(N):
        c.append(centers[i, 0] - radii[i])
        c.append(1.0 - centers[i, 0] - radii[i])
        c.append(centers[i, 1] - radii[i])
        c.append(1.0 - centers[i, 1] - radii[i])
    
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            c.append(dist - radii[i] - radii[j])
    
    # Non-negative radii
    for i in range(N):
        c.append(radii[i])
    
    return np.array(c)


def get_hex_init():
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.04)
    
    rows = [6, 5, 6, 5, 4]
    idx = 0
    for r_idx, sz in enumerate(rows):
        y = 0.12 + r_idx * 0.165
        for c_idx in range(sz):
            if r_idx % 2 == 0:
                x = 0.07 + c_idx * 0.145
            else:
                x = 0.115 + c_idx * 0.145
            centers[idx] = [x, y]
            idx += 1
            if idx >= N:
                break
        if idx >= N:
            break
    
    return centers, radii


def get_grid_init():
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.035)
    
    idx = 0
    for row in range(6):
        for col in range(5):
            if idx < N:
                centers[idx] = [0.1 + col * 0.18, 0.08 + row * 0.16]
                idx += 1
    
    return centers, radii


def get_clustered_init():
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.04)
    
    # Place in clusters
    cluster_centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)]
    idx = 0
    for cx, cy in cluster_centers:
        for dr in range(2):
            for dc in range(3):
                if idx < N:
                    centers[idx] = [cx - 0.08 + dc * 0.08, cy - 0.06 + dr * 0.06]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
        if idx >= N:
            break
    
    return centers, radii


def run_packing():
    np.random.seed(123)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    init_configs = []
    
    # Config 1: Hexagonal
    c, r = get_hex_init()
    init_configs.append((c, r))
    
    # Config 2: Grid
    c, r = get_grid_init()
    init_configs.append((c, r))
    
    # Config 3: Clustered
    c, r = get_clustered_init()
    init_configs.append((c, r))
    
    # Configs 4-8: Perturbed hexagonal
    for _ in range(5):
        c, r = get_hex_init()
        c_pert = c + np.random.randn(*c.shape) * 0.03
        c_pert = np.clip(c_pert, 0.02, 0.98)
        init_configs.append((c_pert, r))
    
    # Configs 9-11: Larger radii start
    for scale in [0.06, 0.07, 0.08]:
        c, r = get_hex_init()
        r = np.full(N, scale)
        init_configs.append((c, r))
    
    bnds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_function}
    
    for centers0, radii0 in init_configs:
        x0 = np.concatenate([centers0.flatten(), radii0])
        
        # Check if initial point is feasible
        init_c = constraint_function(x0)
        if np.min(init_c) < -0.01:
            continue  # Skip infeasible starting points
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bnds,
                constraints=cons,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-15,
                    'disp': False
                }
            )
            
            if result.success:
                s = -result.fun
                final_c = constraint_function(result.x)
                if np.min(final_c) >= -1e-8 and s > best_sum:
                    best_sum = s
                    best_centers = result.x[:2 * N].reshape(N, 2)
                    best_radii = result.x[2 * N:]
        except Exception:
            pass
    
    if best_centers is None:
        centers0, radii0 = get_hex_init()
        best_centers = centers0.copy()
        best_radii = radii0.copy()
        best_sum = np.sum(radii0)
    
    # Post-processing: clean up numerical errors
    best_radii = np.maximum(best_radii, 0.0)
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    # Ensure boundary constraints are satisfied
    for i in range(N):
        r = best_radii[i]
        best_centers[i, 0] = max(r, min(1.0 - r, best_centers[i, 0]))
        best_centers[i, 1] = max(r, min(1.0 - r, best_centers[i, 1]))
    
    # Iterative radius reduction to fix any remaining overlaps
    for _ in range(100):
        max_violation = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                required = best_radii[i] + best_radii[j]
                if dist < required - 1e-10:
                    reduction = (required - dist) / 2 + 1e-8
                    best_radii[i] = max(0, best_radii[i] - reduction)
                    best_radii[j] = max(0, best_radii[j] - reduction)
                    max_violation = max(max_violation, reduction)
        if max_violation < 1e-12:
            break
    
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
