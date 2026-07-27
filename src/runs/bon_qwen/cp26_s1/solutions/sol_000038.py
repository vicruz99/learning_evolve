# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=ddafa521 sum of radii=2.547423 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def _objective(vars):
    return -np.sum(vars[2::3])

def _constraints(vars):
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    c = []
    # Boundary constraints: circle must stay inside [0,1]x[0,1]
    c.extend(xs - rs)
    c.extend(1.0 - xs - rs)
    c.extend(ys - rs)
    c.extend(1.0 - ys - rs)
    
    # Pairwise non-overlap constraints: dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            c.append(np.hypot(dx, dy) - rs[i] - rs[j])
    return np.array(c)

def _jac_objective(vars):
    jac = np.zeros_like(vars)
    jac[2::3] = -1.0
    return jac

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(2025)
    
    # Initialize with a dense hexagonal-like grid
    x_vals = np.linspace(0.12, 0.88, 6)
    y_vals = np.linspace(0.12, 0.88, 5)
    cx, cy = [], []
    count = 0
    for i in range(5):
        for j in range(6):
            if count < N:
                shift = 0.06 if i % 2 == 0 else 0.0
                cx.append(x_vals[j] + shift)
                cy.append(y_vals[i])
                count += 1
        if count >= N:
            break
            
    centers_init = np.array(list(zip(cx, cy)))
    # Add small random perturbation to break symmetry and avoid degeneracy
    centers_init += np.random.uniform(-0.03, 0.03, centers_init.shape)
    centers_init = np.clip(centers_init, 0.05, 0.95)
    radii_init = np.full(N, 0.06)
    
    x0 = np.hstack([centers_init.flatten(), radii_init])
    bnds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N
    
    cons = {'type': 'ineq', 'fun': _constraints}
    
    res = minimize(_objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                   jac=_jac_objective, options={'maxiter': 8000, 'ftol': 1e-12})
    
    xs = res.x[0::3]
    ys = res.x[1::3]
    rs = res.x[2::3]
    
    # Enforce non-negativity and clip coordinates to strictly satisfy validator
    rs = np.maximum(rs, 0.0)
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    
    centers = np.column_stack([xs, ys])
    sum_r = float(np.sum(rs))
    
    return centers, rs, sum_r
