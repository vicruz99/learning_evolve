# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=c18c3d7b sum of radii=2.603894 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def get_pair_indices(n):
    """Precompute indices for all unique circle pairs."""
    i_idx, j_idx = [], []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx), np.array(j_idx)

PAIR_I, PAIR_J = get_pair_indices(N)

def objective(vars_):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(vars_[2*N:])

def constraints(vars_):
    """Compute inequality constraints: boundaries and non-overlap."""
    centers = vars_[:2*N].reshape(N, 2)
    radii = vars_[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = [
        centers[:, 0] - radii,
        1 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1 - centers[:, 1] - radii
    ]
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    c_i = centers[PAIR_I]
    c_j = centers[PAIR_J]
    r_i = radii[PAIR_I]
    r_j = radii[PAIR_J]
    
    dist_sq = np.sum((c_i - c_j)**2, axis=1)
    r_sum_sq = (r_i + r_j)**2
    
    c.append(dist_sq - r_sum_sq)
    return np.concatenate(c)

def generate_init(seed, r_start=0.045):
    """Generate initial center positions and small feasible radii using a hexagonal lattice."""
    np.random.seed(seed)
    centers = []
    y = r_start
    row = 0
    # Generate slightly more points than needed to allow selection
    while len(centers) < N + 5:
        x_start = r_start + (row % 2) * r_start
        x = x_start
        while x <= 1 - r_start:
            centers.append([x, y])
            x += 2 * r_start
        y += np.sqrt(3) * r_start
        row += 1
        
    centers = np.array(centers[:N])
    # Controlled perturbation to break symmetry and ensure feasibility
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, 0.06, 0.94)
    
    radii = np.full(N, r_start)
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Broad multi-start search to explore the landscape
    for seed in range(25):
        x0 = generate_init(seed, r_start=0.045)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-12})
            if -res.fun > best_sum:
                c = res.x[:2*N].reshape(N, 2)
                r = res.x[2*N:]
                # Quick strict validity check
                if np.all(c[:, 0] - r >= -1e-8) and np.all(c[:, 0] + r <= 1 + 1e-8) and \
                   np.all(c[:, 1] - r >= -1e-8) and np.all(c[:, 1] + r <= 1 + 1e-8):
                    dists = np.sqrt(np.sum((c[PAIR_I] - c[PAIR_J])**2, axis=1))
                    if np.all(dists >= r[PAIR_I] + r[PAIR_J] - 1e-8):
                        best_sum = -res.fun
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement from the best configuration found
    if best_x is not None:
        for _ in range(15):
            x0 = best_x.copy()
            # Slight perturbation to escape shallow local minima
            x0[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            x0[:2*N] = np.clip(x0[:2*N], 0.01, 0.99)
            # Slightly reduce radii to guarantee feasibility after perturbation
            x0[2*N:] *= 0.97
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-12})
                if -res.fun > best_sum:
                    c = res.x[:2*N].reshape(N, 2)
                    r = res.x[2*N:]
                    if np.all(c[:, 0] - r >= -1e-8) and np.all(c[:, 0] + r <= 1 + 1e-8) and \
                       np.all(c[:, 1] - r >= -1e-8) and np.all(c[:, 1] + r <= 1 + 1e-8):
                        dists = np.sqrt(np.sum((c[PAIR_I] - c[PAIR_J])**2, axis=1))
                        if np.all(dists >= r[PAIR_I] + r[PAIR_J] - 1e-8):
                            best_sum = -res.fun
                            best_x = res.x.copy()
            except Exception:
                pass

    # Fallback if optimization fails entirely
    if best_x is None:
        best_x = generate_init(0, 0.045)
        
    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    
    # Final strict enforcement to guarantee validator tolerance (1e-12)
    for _ in range(5):
        for i in range(N):
            radii[i] = min(radii[i], centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if radii[i] + radii[j] > d:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    
    return centers, radii, float(np.sum(radii))
