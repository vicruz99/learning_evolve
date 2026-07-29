# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=c3672b16 sum of radii=2.624957 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute boolean mask for upper triangular pairwise constraints
TRI_MASK = np.triu(np.ones((N, N), dtype=bool), k=1)

def objective_func(v):
    """Minimize negative sum of radii to maximize total radius."""
    return -np.sum(v[2*N:])

def constraint_func(v):
    """
    Compute all inequality constraints: g(v) >= 0.
    1. Boundary: r <= x <= 1-r, r <= y <= 1-r
    2. Non-overlap: dist(i,j) >= r_i + r_j
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    # Extract only upper triangle to avoid duplicate constraints
    c = np.concatenate([c, (dist - r_sum)[TRI_MASK]])
    return c

def run_packing():
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_v = None
    
    # Prepare diverse base configurations
    configs = []
    
    # 1. Hexagonal lattice (dense packing prior)
    centers_hex = []
    y = 0.08
    row = 0
    while len(centers_hex) < N:
        x_start = 0.08 if row % 2 == 0 else 0.16
        x = x_start
        while x <= 0.92 and len(centers_hex) < N:
            centers_hex.append([x, y])
            x += 0.16
        y += 0.14
        row += 1
    configs.append(np.array(centers_hex[:N]))
    
    # 2. Uniform grid
    grid = np.array(np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 5))).T.reshape(-1, 2)
    configs.append(grid[:N])
    
    # 3. Random scatter
    np.random.seed(123)
    configs.append(np.random.uniform(0.15, 0.85, size=(N, 2)))
    
    # Multi-start optimization
    for seed in range(30):
        np.random.seed(seed)
        
        base = configs[seed % 3]
        centers = base.copy()
        # Add perturbation to escape local minima and break symmetry
        centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Compute strictly feasible initial radii
        dx = centers[:, 0, None] - centers[None, :, 0]
        dy = centers[:, 1, None] - centers[None, :, 1]
        dist_mat = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_mat, np.inf)
        min_dists = np.min(dist_mat, axis=1)
        
        wall_dists = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                                np.minimum(centers[:, 1], 1 - centers[:, 1]))
        # Start at 90% of the theoretical max to guarantee feasibility
        r_init = np.minimum(min_dists / 2.0, wall_dists) * 0.9
        
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-11, 'disp': False})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                # Verify feasibility before accepting
                if np.all(constraint_func(res.x) >= -1e-6):
                    best_sum = current_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback (should not be reached)
    if best_v is None:
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        return centers, radii, 0.0
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    
    # Strict validity enforcement to handle floating-point drift
    # 1. Enforce boundary constraints
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        
    # 2. Enforce non-overlap constraints with safety margin
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if radii[i] + radii[j] > d:
                shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-8
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)
                
    return centers, radii, float(np.sum(radii))
