# sol_000249 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 01430d11) state=b5d7ec92 sum of radii=2.466275 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(z):
    """Minimize negative sum of radii"""
    return -np.sum(z[2::3])

def constraint_fun(z):
    """
    Computes all inequality constraints g(z) >= 0.
    Returns an array of constraint values.
    """
    num_constr = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    res = np.zeros(num_constr)
    idx = 0
    
    # Boundary constraints: x-r >=0, 1-x-r >=0, y-r >=0, 1-y-r >=0
    for i in range(N_CIRCLES):
        res[idx]   = z[3*i] - z[3*i+2]
        res[idx+1] = 1.0 - z[3*i] - z[3*i+2]
        res[idx+2] = z[3*i+1] - z[3*i+2]
        res[idx+3] = 1.0 - z[3*i+1] - z[3*i+2]
        idx += 4
        
    # Overlap constraints: dist - (r_i + r_j) >= 0
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            dx = z[3*i] - z[3*j]
            dy = z[3*i+1] - z[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            res[idx] = dist - (z[3*i+2] + z[3*j+2])
            idx += 1
    return res

def run_packing():
    # 1. Initialize with a hexagonal grid pattern
    centers_init = []
    step = 0.14  # Spacing slightly larger than 2*initial_radius to ensure feasibility
    for r in range(8):
        for c in range(8):
            x = c * step + (r % 2) * step * 0.5
            y = r * step * np.sqrt(3) / 2
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers_init.append([x, y])
    centers_init = np.array(centers_init[:N_CIRCLES])
    
    # Flatten initial guess: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(N_CIRCLES * 3)
    for i in range(N_CIRCLES):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = 0.06  # Initial radius
        
    # Variable bounds
    bounds = [(0, 1)] * N_CIRCLES + [(0, 1)] * N_CIRCLES + [(1e-6, 0.5)] * N_CIRCLES
    
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    # 2. Run nonlinear optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
                   
    z_opt = res.x
    
    # Extract results
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        centers[i, 0] = z_opt[3*i]
        centers[i, 1] = z_opt[3*i+1]
        radii[i] = z_opt[3*i+2]
        
    # 3. Safety scaling to guarantee validity within 1e-12 tolerance
    k = 0.0
    for i in range(N_CIRCLES):
        r = radii[i]
        # Check boundaries
        if centers[i,0] - r < -1e-12:
            req = (-1e-12 - (centers[i,0]-r))/r
            if req > k: k = req
        if 1.0 - centers[i,0] - r < -1e-12:
            req = (-1e-12 - (1.0 - centers[i,0] - r))/r
            if req > k: k = req
        if centers[i,1] - r < -1e-12:
            req = (-1e-12 - (centers[i,1]-r))/r
            if req > k: k = req
        if 1.0 - centers[i,1] - r < -1e-12:
            req = (-1e-12 - (1.0 - centers[i,1] - r))/r
            if req > k: k = req
            
        # Check overlaps
        for j in range(i+1, N_CIRCLES):
            dist = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
            gap = dist - radii[i] - radii[j]
            if gap < -1e-12:
                req = (-1e-12 - gap) / (radii[i] + radii[j])
                if req > k: k = req
                
    if k > 0:
        radii *= (1.0 - k)
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
