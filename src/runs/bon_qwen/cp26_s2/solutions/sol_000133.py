# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc363b95) state=f2d7145e sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(v, n):
    return -np.sum(v[2*n:])

def _constraints(v, n):
    c = v[:2*n].reshape((n, 2))
    r = v[2*n:]
    cons = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    for i in range(n):
        cons.append(c[i, 0] - r[i])
        cons.append(1.0 - c[i, 0] - r[i])
        cons.append(c[i, 1] - r[i])
        cons.append(1.0 - c[i, 1] - r[i])
        
    # Overlap constraints: dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        xi, yi = c[i]
        ri = r[i]
        for j in range(i + 1, n):
            d = np.hypot(xi - c[j, 0], yi - c[j, 1])
            cons.append(d - ri - r[j])
            
    return np.array(cons)

def _solve(x0, n):
    bnds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': _constraints, 'args': (n,)}
    
    res = minimize(_objective, x0, args=(n,), method='SLSQP', bounds=bnds,
                   constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10})
                   
    centers = res.x[:2*n].reshape((n, 2))
    radii = res.x[2*n:]
    return centers, radii, np.sum(radii)

def run_packing():
    n = 26
    
    # Initialization 1: Hexagonal lattice arrangement
    c1 = np.zeros((n, 2))
    k = 0
    row = 0
    while k < n:
        ncols = 5 if row % 2 == 0 else 4
        y = row * 0.1732
        x0_val = 0.5 - (ncols - 1) * 0.1
        for j in range(ncols):
            if k < n:
                c1[k, 0] = x0_val + j * 0.2
                c1[k, 1] = y
                k += 1
        row += 1
    c1 = np.clip(c1, 0.05, 0.95)
    r1 = np.full(n, 0.08)
    x0_1 = np.concatenate([c1.flatten(), r1])
    
    # Initialization 2: Random perturbation
    np.random.seed(42)
    c2 = np.random.rand(n, 2) * 0.8 + 0.1
    r2 = np.full(n, 0.06)
    x0_2 = np.concatenate([c2.flatten(), r2])
    
    # Run optimization from both starts
    sol1 = _solve(x0_1, n)
    sol2 = _solve(x0_2, n)
    
    # Return the configuration with the higher sum of radii
    return sol1 if sol1[2] > sol2[2] else sol2
