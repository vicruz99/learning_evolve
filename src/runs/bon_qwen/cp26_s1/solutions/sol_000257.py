# sol_000257 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3be09fa9) state=b91457ee sum of radii=2.595543 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

NUM_CIRCLES = 26

def compute_objective(vars):
    """Objective function to minimize: negative sum of radii."""
    return -np.sum(vars[2 * NUM_CIRCLES:])

def compute_constraints(vars):
    """Returns array of constraint values. All must be >= 0."""
    c = []
    n = NUM_CIRCLES
    
    # Boundary constraints
    for i in range(n):
        xi = vars[2 * i]
        yi = vars[2 * i + 1]
        ri = vars[2 * n + i]
        c.append(xi - ri)
        c.append(1.0 - xi - ri)
        c.append(yi - ri)
        c.append(1.0 - yi - ri)
        
    # Overlap constraints (squared distance >= squared sum of radii)
    for i in range(n):
        for j in range(i + 1, n):
            dx = vars[2 * i] - vars[2 * j]
            dy = vars[2 * i + 1] - vars[2 * j + 1]
            d2 = dx * dx + dy * dy
            r_sum = vars[2 * n + i] + vars[2 * n + j]
            c.append(d2 - r_sum * r_sum)
            
    return np.array(c)

def generate_initial_config():
    """Generates a hexagonal lattice packing scaled to fit the unit square."""
    pos = []
    r = 0.105
    dy = np.sqrt(3) * r
    dx = 2.0 * r
    y = r
    row = 0
    
    while len(pos) < NUM_CIRCLES:
        x_off = dx / 2.0 if row % 2 == 1 else 0.0
        x = r + x_off
        while x <= 1.0 - r + 1e-7 and len(pos) < NUM_CIRCLES:
            pos.append([x, y])
            x += dx
        y += dy
        row += 1
        
    pos = np.array(pos[:NUM_CIRCLES])
    rad = np.full(NUM_CIRCLES, r)
    
    # Compute bounding box including radius margins
    min_p = pos.min(axis=0)
    max_p = pos.max(axis=0)
    extent = max_p - min_p + 2.0 * r
    
    # Scale to tightly fit in [0,1]^2
    scale = min(1.0 / extent[0], 1.0 / extent[1])
    
    pos = pos * scale - min_p * scale
    pos += (1.0 - extent * scale) * 0.5
    rad = rad * scale
    
    return pos, rad

def run_packing():
    pos, rad = generate_initial_config()
    x0 = np.concatenate([pos.ravel(), rad])
    
    bnds = [(0.0, 1.0)] * (2 * NUM_CIRCLES) + [(0.0, 0.5)] * NUM_CIRCLES
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    res = opt.minimize(
        compute_objective,
        x0,
        method='SLSQP',
        bounds=bnds,
        constraints=cons,
        options={'maxiter': 4000, 'ftol': 1e-11}
    )
    
    if res.success:
        centers = res.x[:2 * NUM_CIRCLES].reshape(NUM_CIRCLES, 2)
        radii = res.x[2 * NUM_CIRCLES:]
    else:
        centers = pos
        radii = rad
        
    # Post-processing to guarantee strict feasibility
    for i in range(NUM_CIRCLES):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    # Resolve any numerical overlaps
    for _ in range(50):
        max_violation = 0.0
        for i in range(NUM_CIRCLES):
            for j in range(i + 1, NUM_CIRCLES):
                dist = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if dist < req - 1e-9:
                    viol = req - dist
                    if viol > max_violation:
                        max_violation = viol
        if max_violation <= 1e-9:
            break
        radii *= (1.0 - max_violation * 0.1)
        
    return centers, radii, np.sum(radii)
