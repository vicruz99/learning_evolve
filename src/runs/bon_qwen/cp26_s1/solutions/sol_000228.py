# sol_000228 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f043a2e3) state=8e536409 sum of radii=2.620058 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I, J = np.triu_indices(N, k=1)

def objective(z):
    # Minimize negative sum of radii
    return -np.sum(z[2*N:])

def constraints_fun(z):
    c = z[:2*N].reshape(N, 2)
    r = z[2*N:]
    
    # Boundary constraints: x>=r, 1-x>=r, y>=r, 1-y>=r
    b = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Overlap constraints: dist^2 >= (ri+rj)^2
    # Vectorized extraction of coordinates for all pairs
    xi, yi = c[I, 0], c[I, 1]
    xj, yj = c[J, 0], c[J, 1]
    dist_sq = (xi - xj)**2 + (yi - yj)**2
    overlap = dist_sq - (r[I] + r[J])**2
    
    return np.concatenate([b, overlap])

def run_packing():
    best_sum = 0.0
    best_centers = np.zeros((N, 2))
    best_radii = np.zeros(N)
    
    # Variable bounds: centers in [0,1], radii in [1e-7, 0.5]
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_fun}
    
    rng = np.random.default_rng(42)
    
    # Multiple random restarts to escape local minima
    for _ in range(12):
        x0 = rng.random((N, 2))
        r0 = 0.015 * np.ones(N)
        z0 = np.hstack([x0.ravel(), r0])
        
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 2500, 'ftol': 1e-12})
            
            if res.success:
                centers = res.x[:2*N].reshape(N, 2)
                radii = res.x[2*N:]
                s = np.sum(radii)
                
                # Strict validation matching the checker's tolerance
                valid = True
                if np.any(centers[:, 0] < radii - 1e-12) or np.any(centers[:, 0] > 1 - radii + 1e-12): valid = False
                if np.any(centers[:, 1] < radii - 1e-12) or np.any(centers[:, 1] > 1 - radii + 1e-12): valid = False
                
                if valid:
                    dists = np.sqrt((centers[:, None, 0] - centers[None, :, 0])**2 + 
                                    (centers[:, None, 1] - centers[None, :, 1])**2)
                    if np.any(dists[I, J] < radii[I] + radii[J] - 1e-12):
                        valid = False
                        
                if valid and s > best_sum:
                    best_sum = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
        except Exception:
            continue
            
    # Fallback configuration if optimization fails
    if best_sum < 1.0:
        gs = 5
        step = 1.0 / (gs + 1)
        coords = np.array([[step*(k%gs+1), step*(k//gs+1)] for k in range(25)] + [[0.5, 0.05]])
        best_centers = coords
        best_radii = np.full(N, step/2 * 0.98)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
