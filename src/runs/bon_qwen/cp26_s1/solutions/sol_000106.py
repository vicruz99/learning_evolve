# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 81b841bb) state=a433dbd3 sum of radii=2.612832 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def _objective(vars, N):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(vars[2*N:])

def _jac(vars, N):
    """Gradient of the objective function"""
    jac = np.zeros_like(vars)
    jac[2*N:] = -1.0
    return jac

def _constraints(vars, N):
    """Constraint functions: boundaries and non-overlap"""
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    n_cons = N*4 + N*(N-1)//2
    cons = np.empty(n_cons)
    
    # Boundary constraints: circles inside [0,1]^2
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        cons[4*i] = x - r
        cons[4*i+1] = 1.0 - x - r
        cons[4*i+2] = y - r
        cons[4*i+3] = 1.0 - y - r
        
    # Pairwise non-overlap constraints
    idx = 4*N
    for i in range(N):
        ci = centers[i]
        ri = radii[i]
        for j in range(i+1, N):
            cj = centers[j]
            rj = radii[j]
            d = np.sqrt(np.sum((ci - cj)**2))
            cons[idx] = d - ri - rj
            idx += 1
    return cons

def _get_init_config():
    """Generate a hexagonal lattice initial configuration"""
    centers = []
    # Create a staggered grid
    for r in range(6):
        for c in range(5):
            if len(centers) >= N_CIRCLES:
                break
            x = 0.1 + c*0.2 + (r%2)*0.1
            y = 0.1 + r*0.17
            centers.append([x, y])
            
    centers = np.array(centers)
    # Normalize to fit comfortably inside the unit square
    min_c = centers.min(axis=0)
    max_c = centers.max(axis=0)
    centers = (centers - min_c) * (0.85 / (max_c - min_c)) + 0.05
    centers = np.clip(centers, 0.05, 0.95)
    
    # Small initial radii to ensure strict feasibility for the solver
    radii = np.full(N_CIRCLES, 0.01)
    return centers, radii

def _optimize_once(centers, radii, N):
    """Run SLSQP optimization from a given configuration"""
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * 2*N + [(0.001, 0.5)] * N
    cons = {'type': 'ineq', 'fun': _constraints, 'args': (N,)}
    
    res = minimize(_objective, x0, args=(N,), method='SLSQP', 
                   jac=_jac, bounds=bounds, constraints=cons,
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
    return res.x

def run_packing():
    N = N_CIRCLES
    best_sum = -np.inf
    best_x = None
    
    init_c, init_r = _get_init_config()
    
    # Multi-restart strategy to escape local minima
    for seed in range(5):
        np.random.seed(seed)
        # Perturb positions and radii slightly
        c_p = init_c + np.random.uniform(-0.015, 0.015, init_c.shape)
        c_p = np.clip(c_p, 0.05, 0.95)
        r_p = init_r + np.random.uniform(-0.005, 0.005, N)
        r_p = np.clip(r_p, 0.005, 0.15)
        
        res_x = _optimize_once(c_p, r_p, N)
        curr_sum = np.sum(res_x[2*N:])
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_x = res_x
            
    opt_centers = best_x[:2*N].reshape(N, 2)
    opt_radii = best_x[2*N:]
    
    # Apply small safety margin to guarantee validity against numerical precision
    opt_radii *= 0.998
    
    return opt_centers, opt_radii, float(np.sum(opt_radii))
