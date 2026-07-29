# sol_000194 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bafdbd7e) state=ea575f9e sum of radii=1.394535 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(z):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(z[2::3])

def constraints(z):
    """Constraint function: boundaries and non-overlap"""
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Boundary constraints: circles must be inside [0,1]x[0,1]
    c_boundary = np.concatenate([
        x - r,           # x >= r
        1.0 - x - r,     # x <= 1 - r
        y - r,           # y >= r
        1.0 - y - r      # y <= 1 - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    c_pair = np.array([], dtype=float)
    for i in range(N_CIRCLES):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dr = r[i] + r[i+1:]
        c_pair = np.concatenate([c_pair, dx**2 + dy**2 - dr**2])
        
    return np.concatenate([c_boundary, c_pair])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initial Configuration: Hexagonal packing arrangement
    r0 = 0.04
    dx = 2 * r0 * 1.2
    dy = np.sqrt(3) * r0 * 1.2
    
    # Row counts summing to 26: 5 + 6 + 5 + 6 + 4 = 26
    row_counts = [5, 6, 5, 6, 4]
    centers_init = []
    yy = 0.2
    
    for i, cnt in enumerate(row_counts):
        row_width = (cnt - 1) * dx
        xx = (1.0 - row_width) / 2.0 + r0
        # Offset every other row for hexagonal pattern
        if i % 2 == 1:
            xx += dx / 2.0
        for _ in range(cnt):
            centers_init.append([xx, yy])
            xx += dx
        yy += dy
        
    # 2. Prepare initial vector for optimizer
    z0 = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        z0[3*i] = centers_init[i][0]
        z0[3*i+1] = centers_init[i][1]
        z0[3*i+2] = r0
        
    # Deterministic perturbation to break symmetry and avoid local traps
    for i in range(N_CIRCLES):
        z0[3*i] += (i % 11) * 0.002
        z0[3*i+1] += (i % 7) * 0.002
        
    # 3. Define bounds and constraints
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    # 4. Run Optimization
    res = minimize(
        objective, z0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
    )
    
    z_opt = res.x
    centers = np.column_stack((z_opt[0::3], z_opt[1::3]))
    radii = z_opt[2::3]
    
    # 5. Post-processing: Ensure strict validity within numerical tolerance
    # Clamp radii to boundary constraints
    for i in range(N_CIRCLES):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        radii[i] = min(radii[i], max_r - 1e-13)
        
    # Ensure pairwise separation
    # Iteratively shrink radii if overlaps persist due to numerical precision
    for _ in range(5): # Few passes to resolve any tight overlaps
        changed = False
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dist = np.sqrt((centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2)
                required = radii[i] + radii[j]
                if dist < required - 1e-12:
                    # Shrink both radii slightly to satisfy constraint
                    scale = dist / required
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
