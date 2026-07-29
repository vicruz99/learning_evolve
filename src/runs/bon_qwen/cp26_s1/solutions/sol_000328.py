# sol_000328 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fe3e1745) state=a3431e4a sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

# Constants
N_CIRCLES = 26
PENALTY = 10000000.0

def objective_lbgfsb(vars):
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        centers[i] = [vars[3*i], vars[3*i+1]]
        radii[i] = vars[3*i+2]
        
    obj = -np.sum(radii)
    penalty = 0.0
    
    for i in range(N_CIRCLES):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: penalty += PENALTY * (x - r)**2
        if 1 - x - r < 0: penalty += PENALTY * (1 - x - r)**2
        if y - r < 0: penalty += PENALTY * (y - r)**2
        if 1 - y - r < 0: penalty += PENALTY * (1 - y - r)**2
        
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                diff = sum_r - dist
                penalty += PENALTY * diff**2
                
    return obj + penalty

def get_hex_init():
    r = 0.09
    pts = []
    y = r
    for row in range(8):
        x = r
        if row % 2 == 1:
            x += r
        limit = 5 if row % 2 == 0 else 4
        count = 0
        while count < limit and x + r <= 1.0:
            pts.append([x, y])
            x += 2 * r
            count += 1
        y += r * math.sqrt(3)
        if len(pts) >= N_CIRCLES:
            break
    return pts[:N_CIRCLES]

def obj_slsqp(vars):
    s = 0.0
    for i in range(N_CIRCLES):
        s += vars[3*i+2]
    return -s

def cons_boundary(vars):
    c = []
    for i in range(N_CIRCLES):
        x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
        c.extend([x - r, 1 - x - r, y - r, 1 - y - r])
    return np.array(c)

def cons_overlap(vars):
    c = []
    for i in range(N_CIRCLES):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        for j in range(i+1, N_CIRCLES):
            xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            c.append(dist - ri - rj)
    return np.array(c)

def run_packing():
    inits = [get_hex_init()]
    for _ in range(2):
        pts = []
        for _ in range(N_CIRCLES):
            pts.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)])
        inits.append(pts)
        
    best_sum = -1.0
    best_centers = None
    best_radii = None

    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])

    # Step 1: Global search with L-BFGS-B (penalty method)
    for pts in inits:
        x0 = []
        for p in pts:
            x0.extend([p[0], p[1], 0.08])
        
        try:
            res = minimize(objective_lbgfsb, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            centers = np.zeros((N_CIRCLES, 2))
            radii = np.zeros(N_CIRCLES)
            for i in range(N_CIRCLES):
                centers[i] = [res.x[3*i], res.x[3*i+1]]
                radii[i] = res.x[3*i+2]
            
            current_sum = np.sum(radii)
            
            # Use this as a candidate for refinement
            # We don't strictly require it to be valid here, just good sum
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        except:
            pass

    # Step 2: Refine best candidate with SLSQP (strict constraints)
    if best_centers is not None:
        x0_slsqp = []
        for i in range(N_CIRCLES):
            x0_slsqp.extend([best_centers[i,0], best_centers[i,1], best_radii[i]])
            
        constraints = [
            {'type': 'ineq', 'fun': cons_boundary},
            {'type': 'ineq', 'fun': cons_overlap}
        ]
        
        try:
            res_slsqp = minimize(obj_slsqp, x0_slsqp, method='SLSQP', 
                                 bounds=bounds, constraints=constraints,
                                 options={'maxiter': 2000, 'ftol': 1e-12})
            
            sum_r = 0.0
            for i in range(N_CIRCLES):
                sum_r += res_slsqp.x[3*i+2]
            
            if sum_r > best_sum:
                best_sum = sum_r
                best_centers = np.zeros((N_CIRCLES, 2))
                best_radii = np.zeros(N_CIRCLES)
                for i in range(N_CIRCLES):
                    best_centers[i] = [res_slsqp.x[3*i], res_slsqp.x[3*i+1]]
                    best_radii[i] = res_slsqp.x[3*i+2]
        except:
            pass

    # Fallback if everything failed
    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))
        best_radii = np.full(N_CIRCLES, 0.02)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(np.sum(best_radii))
