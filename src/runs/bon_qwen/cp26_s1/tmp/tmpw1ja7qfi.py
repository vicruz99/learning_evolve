import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def objective(v):
    # Maximize common radius r => minimize -r
    return -v[-1]

def boundary_constraints(v):
    # Ensure circles stay inside [0,1]x[0,1]
    r = v[-1]
    cs = v[:2*N].reshape(N, 2)
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    return np.concatenate([cs[:, 0] - r, 1.0 - cs[:, 0] - r, cs[:, 1] - r, 1.0 - cs[:, 1] - r])

def pairwise_constraints(v):
    # Ensure no overlaps: dist(i,j) >= 2r
    r = v[-1]
    cs = v[:2*N].reshape(N, 2)
    dists = cdist(cs, cs)
    i, j = np.tril_indices(N, -1)
    return dists[i, j] - 2.0 * r

def run_packing():
    np.random.seed(42)
    
    # Initialize with a perturbed 5x5 grid + 1 center circle
    pts = np.linspace(0.15, 0.85, 5)
    grid_x, grid_y = np.meshgrid(pts, pts)
    centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    centers = np.vstack([centers, [0.5, 0.5]])
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    
    r_init = 0.08
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.2)]
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': pairwise_constraints}
    ]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
                   
    # Apply tiny safety margin to guarantee strict feasibility within validator tolerance
    best_r = res.x[-1] * 0.9999
    best_centers = res.x[:2*N].reshape(N, 2)
    
    return best_centers, np.full(N, best_r), N * best_r