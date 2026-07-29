# sol_000054 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=2b2a3d30 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    N = 26
    # Precompute indices for the upper triangle (pairwise constraints)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def objective(v):
        """Minimize negative sum of radii -> Maximize sum of radii"""
        return -np.sum(v[2*N:])

    def constraints(v):
        """Compute all inequality constraints in vectorized form.
        Returns array where each element >= 0 indicates satisfied constraint."""
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        cons = np.empty(4*N + len(i_idx))
        
        # Boundary constraints: x >= r, x+r <= 1, y >= r, y+r <= 1
        cons[:N] = x - r
        cons[N:2*N] = 1.0 - x - r
        cons[2*N:3*N] = y - r
        cons[3*N:4*N] = 1.0 - y - r
        
        # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
        # Using squared distance avoids sqrt singularities and is smoother for SLSQP
        X = x[i_idx] - x[j_idx]
        Y = y[i_idx] - y[j_idx]
        R = r[i_idx] + r[j_idx]
        cons[4*N:] = X*X + Y*Y - R*R
        
        return cons

    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    def try_optimize(v0):
        nonlocal best_sum, best_v
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            pass

    # 1. Structured initial configurations
    configs = []
    
    # Hexagonal lattice packing
    r_h = 0.09
    y_h = r_h
    row_h = 0
    pts_hex = []
    while len(pts_hex) < N:
        x_h = r_h if row_h % 2 == 0 else 2*r_h
        while x_h <= 1 - r_h and len(pts_hex) < N:
            pts_hex.append((x_h, y_h))
            x_h += 2*r_h
        y_h += r_h * np.sqrt(3)
        row_h += 1
    configs.append(np.array(pts_hex[:N]))
    
    # Dense grid packing
    pts_grid = []
    for i in range(5):
        for j in range(5):
            pts_grid.append((0.1 + i*0.2, 0.1 + j*0.2))
    pts_grid.append((0.5, 0.5))  # Extra circle in center
    configs.append(np.array(pts_grid[:N]))
    
    # Random dense packing
    np.random.seed(42)
    configs.append(np.random.uniform(0.1, 0.9, size=(N, 2)))
    
    # Optimize from each structured start
    for cfg in configs:
        v0 = np.concatenate([cfg[:, 0], cfg[:, 1], np.full(N, 0.04)])
        try_optimize(v0)
        
    # 2. Basin hopping / local perturbation search
    np.random.seed(123)
    for _ in range(60):
        if best_v is not None and np.random.rand() < 0.75:
            # Perturb best known configuration
            v0 = best_v.copy()
            v0[:2*N] += np.random.uniform(-0.015, 0.015, size=2*N)
            v0[2*N:] += np.random.uniform(-0.005, 0.005, size=N)
        else:
            # Random restart to explore new basins
            v0 = np.random.uniform(0.05, 0.95, size=2*N)
            v0 = np.concatenate([v0, np.full(N, 0.03)])
            
        v0 = np.clip(v0, [0.0]*2*N + [0.0]*N, [1.0]*2*N + [0.5]*N)
        try_optimize(v0)
        
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    
    # 3. Strict feasibility enforcement
    # Iteratively shrink radii to guarantee validator tolerance is met
    for _ in range(5):
        changed = False
        for i in range(N):
            # Boundary constraints
            lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > lim + 1e-9:
                radii[i] = lim
                changed = True
            # Pairwise constraints
            for j in range(N):
                if i != j:
                    d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    lim = d - radii[j]
                    if radii[i] > lim + 1e-9:
                        radii[i] = max(0.0, lim)
                        changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
