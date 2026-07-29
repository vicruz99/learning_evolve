# sol_000202 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 263f0241) state=1e039abd sum of radii=2.479803 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import warnings

warnings.filterwarnings('ignore')

N_CIRCLES = 26

def objective_func(x):
    """Objective: maximize radius r (minimize -r)"""
    return -x[-1]

def bound_constraints_func(x):
    """Boundary constraints: circles must stay inside [0,1]x[0,1]"""
    centers = x[:2*N_CIRCLES].reshape((N_CIRCLES, 2))
    r = x[-1]
    con = np.empty(4 * N_CIRCLES)
    for i in range(N_CIRCLES):
        con[4*i]   = centers[i, 0] - r
        con[4*i+1] = 1.0 - centers[i, 0] - r
        con[4*i+2] = centers[i, 1] - r
        con[4*i+3] = 1.0 - centers[i, 1] - r
    return con

def pair_constraints_func(x):
    """Pairwise non-overlap constraints: dist^2 >= 4r^2"""
    centers = x[:2*N_CIRCLES].reshape((N_CIRCLES, 2))
    r = x[-1]
    # Vectorized pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Extract upper triangle to avoid duplicates and self-comparisons
    idxs = np.triu_indices(N_CIRCLES, k=1)
    return dist_sq[idxs] - 4.0 * r**2

def generate_initial_config(seed=42):
    """Generate a structured initial layout with slight random perturbation"""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N_CIRCLES, 2))
    cols = 6
    for i in range(N_CIRCLES):
        r_idx = i // cols
        c_idx = i % cols
        centers[i, 0] = (c_idx + 0.5) / cols
        centers[i, 1] = (r_idx + 0.5) / 5.0
    # Perturb to break symmetry and help optimizer explore
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    # Ensure initial positions are safely inside boundaries
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing():
    # Variable bounds: centers in [0,1], radius in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(1e-6, 0.5)]
    
    # Define constraints
    bnds_con = NonlinearConstraint(bound_constraints_func, 0, np.inf)
    pairs_con = NonlinearConstraint(pair_constraints_func, 0, np.inf)
    
    best_r = 0.0
    best_centers = None
    
    # Try multiple starting configurations to improve global search
    initial_configs = [generate_initial_config(seed=i) for i in range(3)]
    
    for centers0 in initial_configs:
        x0 = np.concatenate([centers0.flatten(), [0.08]])
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints=[bnds_con, pairs_con],
                           options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
            current_r = -res.fun
            if current_r > best_r:
                best_r = current_r
                best_centers = res.x[:2*N_CIRCLES].reshape((N_CIRCLES, 2))
        except Exception:
            continue
            
    # Fallback to a valid configuration if optimization yields nothing useful
    if best_centers is None or best_r < 0.05:
        best_centers = generate_initial_config(seed=0)
        best_r = 0.08
        
    # Ensure numerical validity
    best_r = max(best_r, 1e-7)
    for i in range(N_CIRCLES):
        best_centers[i, 0] = np.clip(best_centers[i, 0], best_r, 1.0 - best_r)
        best_centers[i, 1] = np.clip(best_centers[i, 1], best_r, 1.0 - best_r)
        
    radii = np.full(N_CIRCLES, best_r)
    return best_centers, radii, np.sum(radii)
