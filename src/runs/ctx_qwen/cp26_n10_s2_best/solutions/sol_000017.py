# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b9ac6cc) state=e5afe9d6 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def obj(vars_):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars_[2*N:])

def cons_overlap(vars_):
    """Non-overlap constraints: dist^2 >= (r1 + r2)^2"""
    centers = vars_[:2*N].reshape((N, 2))
    radii = vars_[2*N:]
    con = []
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx**2 + dy**2
            r_sum = radii[i] + radii[j]
            con.append(dist_sq - r_sum**2)
    return np.array(con)

def cons_boundary(vars_):
    """Boundary constraints: circles inside [0,1]^2"""
    centers = vars_[:2*N].reshape((N, 2))
    radii = vars_[2*N:]
    con = []
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        con.append(x - r)
        con.append(1 - (x + r))
        con.append(y - r)
        con.append(1 - (y + r))
    return np.array(con)

def get_initial_guess(seed):
    """Generate a hexagonal lattice initialization with perturbation"""
    np.random.seed(seed)
    centers = []
    r_init = 0.08
    y = r_init
    row = 0
    while len(centers) < N:
        x_start = r_init if row % 2 == 0 else 2 * r_init
        x = x_start
        while x <= 1 - r_init and len(centers) < N:
            centers.append([x, y])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
        
    centers = np.array(centers[:N])
    # Break symmetry with small random noise
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Start with a feasible radius
    radii = np.full(N, 0.05)
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    constraints = [{'type': 'ineq', 'fun': cons_overlap},
                   {'type': 'ineq', 'fun': cons_boundary}]
    
    best_sum = -1.0
    best_x = None
    
    # Multi-start optimization to avoid poor local minima
    for seed in range(5):
        try:
            x0 = get_initial_guess(seed)
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_x = res.x.copy()
        except Exception:
            continue
            
    centers = best_x[:2*N].reshape((N, 2))
    radii = best_x[2*N:]
    
    # Final safety clamp to guarantee validator tolerance is met
    # SLSQP ftol=1e-12 may leave constraints exactly on the edge.
    # A negligible shrink ensures strict inequality for the 1e-12 check.
    radii *= 0.9999999995
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    
    return centers, radii, np.sum(radii)
