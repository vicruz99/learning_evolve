# sol_000102 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 15bab5cf) state=018fa271 sum of radii=2.591698 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_fun(vars_):
    # Negative sum of radii to minimize
    return -np.sum(vars_[2::3])

def obj_jac(vars_):
    # Analytical gradient of the objective
    jac = np.zeros_like(vars_)
    jac[2::3] = -1.0
    return jac

def make_constraints(n):
    cons = []
    # Boundary constraints for each circle
    for i in range(n):
        def c_x(v, i=i): return v[3*i] - v[3*i+2]
        def c_1x(v, i=i): return 1.0 - v[3*i] - v[3*i+2]
        def c_y(v, i=i): return v[3*i+1] - v[3*i+2]
        def c_1y(v, i=i): return 1.0 - v[3*i+1] - v[3*i+2]
        cons.extend([
            {'type': 'ineq', 'fun': c_x},
            {'type': 'ineq', 'fun': c_1x},
            {'type': 'ineq', 'fun': c_y},
            {'type': 'ineq', 'fun': c_1y}
        ])
    # Non-overlap constraints for each pair
    for i in range(n):
        for j in range(i+1, n):
            def c_ov(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dr = v[3*i+2] + v[3*j+2]
                return dx*dx + dy*dy - dr*dr
            cons.append({'type': 'ineq', 'fun': c_ov})
    return cons

def run_packing():
    n = 26
    # Initialize with a 5x5 grid plus one in the center
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.2, 0.1 + j * 0.2]
            idx += 1
    centers[25] = [0.5, 0.5]
    
    # Add small random perturbation to break symmetry and avoid deadlocks
    np.random.seed(42)
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n, 0.08)
    
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    constraints = make_constraints(n)
    
    # Run SLSQP optimization
    result = minimize(obj_fun, x0, method='SLSQP', jac=obj_jac, 
                      bounds=bounds, constraints=constraints, 
                      options={'maxiter': 3000, 'ftol': 1e-12})
                      
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i, 0] = result.x[3*i]
        final_centers[i, 1] = result.x[3*i+1]
        final_radii[i] = max(0.0, result.x[3*i+2])
        
    # Post-processing: guarantee strict validity by clipping radii if needed
    for i in range(n):
        r_lim = min(final_centers[i,0], 1-final_centers[i,0], 
                    final_centers[i,1], 1-final_centers[i,1])
        for j in range(n):
            if i == j: continue
            dist = np.hypot(final_centers[i,0]-final_centers[j,0], 
                            final_centers[i,1]-final_centers[j,1])
            r_lim = min(r_lim, dist - final_radii[j])
        final_radii[i] = max(0.0, r_lim)
        
    return final_centers, final_radii, np.sum(final_radii)
