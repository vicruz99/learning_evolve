# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4a327247) state=d8b66414 sum of radii=2.617760 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Minimize negative sum of radii (equivalent to maximizing sum)."""
    radii = vars[2*N:]
    return -np.sum(radii)

def constraint_fun(vars):
    """Vectorized constraint function: boundary and non-overlap."""
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    cons = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    cons.append(centers[:, 0] - radii)
    cons.append(1 - radii - centers[:, 0])
    cons.append(centers[:, 1] - radii)
    cons.append(1 - radii - centers[:, 1])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = centers[:, 0, None] - centers[:, 0]
    dy = centers[:, 1, None] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    r_sum = radii[:, None] + radii[None, :]
    # Only consider upper triangle pairs to avoid duplicates
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    cons.append((dist_sq - r_sum**2)[mask])
    
    return np.concatenate(cons)

def run_packing():
    np.random.seed(42)
    
    # Initialize centers on a hexagonal lattice pattern
    pts = []
    s = 0.22
    for i in range(7):
        for j in range(7):
            x = 0.1 + j * s + (0.5 * s if i % 2 == 1 else 0)
            y = 0.1 + i * s * np.sqrt(3)/2
            if x <= 0.9 and y <= 0.9:
                pts.append([x, y])
                
    pts = np.array(pts[:N])
    # Fallback to grid if hex pattern yields fewer than N points
    if len(pts) < N:
        grid = np.array(np.meshgrid(np.linspace(0.15, 0.85, 6), np.linspace(0.15, 0.85, 6))).T.reshape(-1, 2)
        pts = np.vstack([pts, grid[len(pts):N]])
        
    # Small perturbation to break symmetry and aid optimization
    pts += np.random.uniform(-0.005, 0.005, pts.shape)
    
    # Start with small radii to ensure initial feasibility
    radii_init = np.full(N, 0.03)
    x0 = np.hstack([pts.flatten(), radii_init])
    
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    # Run SLSQP optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
                   
    centers = res.x[:2*N].reshape(N, 2)
    radii = res.x[2*N:]
    
    # Ensure numerical cleanliness
    radii = np.maximum(radii, 0.0)
    centers = np.clip(centers, 0.0, 1.0)
    
    return centers, radii, float(np.sum(radii))
