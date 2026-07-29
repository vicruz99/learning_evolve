# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 028484b6) state=e615ce1b sum of radii=2.591663 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N_CIRCLES = 26

def objective_func(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2*N_CIRCLES:])

def constraint_func(x):
    """Evaluate inequality constraints: >= 0 means feasible."""
    centers = x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = x[2*N_CIRCLES:]
    
    # Boundary constraints: 4*N
    b = np.concatenate([
        centers[:, 0] - radii,
        1.0 - (centers[:, 0] + radii),
        centers[:, 1] - radii,
        1.0 - (centers[:, 1] + radii)
    ])
    
    # Pairwise non-overlap constraints: N*(N-1)/2
    dists = pdist(centers)
    # Construct sum of radii for all upper-triangle pairs matching pdist order
    r_sum = (radii[:, None] + radii[None, :])[np.triu_indices(N_CIRCLES, k=1)]
    
    return np.concatenate([b, dists - r_sum])

def run_packing():
    # 1. Generate initial hexagonal configuration
    centers = []
    r_init = 0.12
    y = r_init
    while y + r_init <= 1.0 and len(centers) < N_CIRCLES:
        x = r_init
        row_idx = int((y - r_init) / (r_init * np.sqrt(3.0)))
        offset = r_init if row_idx % 2 == 1 else 0.0
        while x + r_init <= 1.0 and len(centers) < N_CIRCLES:
            centers.append([x + offset, y])
            x += 2.0 * r_init
        y += r_init * np.sqrt(3.0)
        
    # Fill remaining if grid was sparse
    while len(centers) < N_CIRCLES:
        centers.append([np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1])
        
    centers = np.array(centers[:N_CIRCLES])
    # Add small perturbation to break symmetry
    centers += np.random.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    # 2. Compute feasible initial radii based on closest distances
    min_gap = 1.0
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            d = np.sqrt(np.sum((centers[i]-centers[j])**2))
            if d < min_gap: min_gap = d
        for c in centers[i]:
            if c < min_gap: min_gap = c
            if 1.0-c < min_gap: min_gap = 1.0-c
            
    radii = np.full(N_CIRCLES, min_gap / 2.0 * 0.85)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # 3. Setup bounds and constraints
    bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # 4. Run optimization
    res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                   
    final_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    final_radii = res.x[2*N_CIRCLES:]
    total_sum = np.sum(final_radii)
    
    # 5. Post-optimization safety check
    constraints_vals = constraint_func(res.x)
    if np.min(constraints_vals) < -1e-9:
        # Slightly shrink radii to guarantee strict validity against tolerance
        scale = 1.0 - 2e-5
        final_radii *= scale
        total_sum = np.sum(final_radii)
        
    return final_centers, final_radii, total_sum
