# sol_000001 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a5b5ea2) state=9f980ef0 sum of radii=2.489205 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def packing_objective(vars):
    """Objective function: maximize radius (minimize negative radius)."""
    return -vars[2 * N_CIRCLES]

def packing_constraints(vars):
    """
    Constraint function: returns inequality constraints >= 0.
    Enforces boundary containment and non-overlap.
    """
    cs = vars[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = vars[2 * N_CIRCLES]
    
    # Boundary constraints: r <= x, y <= 1-r
    lb = cs - r
    ub = 1.0 - cs - r
    
    # Pairwise non-overlap constraints: distance >= 2r
    dx = cs[:, 0, np.newaxis] - cs[:, 0]
    dy = cs[:, 1, np.newaxis] - cs[:, 1]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, 0)
    
    # Extract lower triangular part to check each pair exactly once
    lower_tri = dists[np.tril_indices(N_CIRCLES, k=-1)]
    pairwise = lower_tri - 2 * r
    
    return np.concatenate([lb.flatten(), ub.flatten(), pairwise])

def run_packing():
    inits = []
    
    # Initial Configuration 1: 5x5 grid + 1 in center
    c1 = np.zeros((N_CIRCLES, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            c1[idx] = [0.1 * (i + 0.5), 0.1 * (j + 0.5)]
            idx += 1
    c1[-1] = [0.5, 0.5]
    inits.append(c1)
    
    # Initial Configuration 2: Hexagonal lattice
    c2 = np.zeros((N_CIRCLES, 2))
    idx = 0
    y = 0.09
    row_idx = 0
    while idx < N_CIRCLES:
        shift = 0.09 if row_idx % 2 == 1 else 0
        x = 0.09 + shift
        while x + 0.09 <= 1.0 and idx < N_CIRCLES:
            c2[idx] = [x, y]
            x += 0.18
            idx += 1
        y += 0.09 * np.sqrt(3)
        row_idx += 1
    inits.append(c2)
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    bounds = [(0, 1) for _ in range(2 * N_CIRCLES)] + [(0, 0.5)]
    constraints = {'type': 'ineq', 'fun': packing_constraints}
    
    for init in inits:
        for _ in range(8):
            # Random perturbation to escape local minima
            c_pert = init + np.random.randn(N_CIRCLES, 2) * 0.015
            c_pert = np.clip(c_pert, 0.05, 0.95)
            r_init = 0.09 + np.random.randn() * 0.005
            if r_init < 0.01:
                r_init = 0.01
                
            x0 = np.concatenate([c_pert.flatten(), [r_init]])
            
            try:
                res = minimize(packing_objective, x0, method='SLSQP', 
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 1000, 'ftol': 1e-10})
                
                if res.success:
                    r_opt = res.x[-1]
                    s = N_CIRCLES * r_opt
                    if s > best_sum:
                        best_sum = s
                        best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
                        best_radii = np.full(N_CIRCLES, r_opt)
            except Exception:
                continue
                
    if best_centers is None:
        # Fallback to a valid, though suboptimal, packing
        best_centers = np.array([[0.1*(i+0.5), 0.1*(j+0.5)] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        best_radii = np.full(N_CIRCLES, 0.09)
        best_sum = N_CIRCLES * 0.09
        
    return best_centers, best_radii, best_sum
