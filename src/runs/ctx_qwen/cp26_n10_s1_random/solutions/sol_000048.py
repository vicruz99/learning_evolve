# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=2ce766a2 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def compute_constraints(vars, n):
    """Constraint function: returns array of values >= 0 for valid packing."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    boundary_cons = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints (squared distance)
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    dist_sq = dx**2 + dy**2
    r_sum_sq = dr**2
    
    rows, cols = np.triu_indices(n, k=1)
    overlap_cons = dist_sq[rows, cols] - r_sum_sq[rows, cols]
    
    return np.concatenate([boundary_cons, overlap_cons])

def run_packing():
    n = 26
    best_sum = -1.0
    best_result = None
    
    # Bounds for [x, y, r] for each circle
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * n
    
    constraint_dict = {
        'type': 'ineq',
        'fun': compute_constraints,
        'args': (n,)
    }
    
    np.random.seed(42)
    
    # Generate initial hexagonal lattice configuration
    r_init = 0.095
    pts = []
    y = r_init
    row = 0
    while len(pts) < n:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x <= 1.0 - r_init and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
        
    pts = np.array(pts[:n])
    
    # Prepare multiple starting configurations
    inits = []
    
    # 1. Base hexagonal lattice
    v = np.zeros(3 * n)
    v[0::3] = pts[:, 0]
    v[1::3] = pts[:, 1]
    v[2::3] = r_init
    inits.append(v)
    
    # 2. Perturbed hexagonal lattices to explore local minima
    for _ in range(5):
        noise = np.random.uniform(-0.02, 0.02, size=(n, 2))
        p = np.clip(pts + noise, 0.05, 0.95)
        v = np.zeros(3 * n)
        v[0::3] = p[:, 0]
        v[1::3] = p[:, 1]
        v[2::3] = r_init
        inits.append(v)
        
    # 3. Random starts with small radii to ensure feasibility
    for _ in range(3):
        v = np.zeros(3 * n)
        v[0::3] = np.random.uniform(0.1, 0.9, n)
        v[1::3] = np.random.uniform(0.1, 0.9, n)
        v[2::3] = 0.04 * np.ones(n)
        inits.append(v)
        
    # Run optimization for each configuration
    for init_v in inits:
        try:
            res = minimize(
                compute_objective,
                init_v,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_dict,
                options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                r_opt = res.x[2::3]
                current_sum = np.sum(r_opt)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res
        except Exception:
            continue
            
    # Fallback if optimization fails (highly unlikely)
    if best_result is None:
        centers = pts
        radii = np.full(n, r_init)
        return centers, radii, np.sum(radii)
        
    # Extract optimal centers and radii
    centers = np.column_stack((best_result.x[0::3], best_result.x[1::3]))
    radii = best_result.x[2::3]
    
    # Final safety adjustment: slightly shrink radii if any constraint is marginally violated
    # to strictly satisfy the validator's 1e-12 tolerance due to numerical precision.
    def is_valid(c, r):
        for i in range(n):
            x, y = c[i]
            ri = r[i]
            if x - ri < -1e-12 or x + ri > 1 + 1e-12 or y - ri < -1e-12 or y + ri > 1 + 1e-12:
                return False
            for j in range(i + 1, n):
                dist = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                if dist < r[i] + r[j] - 1e-12:
                    return False
        return True

    if not is_valid(centers, radii):
        scale = 0.9999
        while not is_valid(centers, radii):
            radii *= scale
            if scale < 0.999999: break
            scale *= 0.9999999
            
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
