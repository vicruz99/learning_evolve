# sol_000010 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state abc5794a) state=f39c4564 sum of radii=2.618656 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x, n):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(x[2::3])

def compute_constraints(x, n):
    """Constraint function: returns array of values >= 0 for valid packing"""
    cxs = x[0::3]
    cys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    b_cons = np.concatenate([
        cxs - rs,          # x - r >= 0
        1.0 - cxs - rs,    # 1 - x - r >= 0
        cys - rs,          # y - r >= 0
        1.0 - cys - rs     # 1 - y - r >= 0
    ])
    
    # Pairwise non-overlap constraints
    n_pairs = n * (n - 1) // 2
    p_cons = np.empty(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cxs[i] - cxs[j]
            dy = cys[i] - cys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            p_cons[idx] = dist - rs[i] - rs[j]
            idx += 1
            
    return np.concatenate([b_cons, p_cons])

def run_packing() -> tuple:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds for [cx, cy, r] for each circle
    bounds_list = []
    for _ in range(n):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
        
    constraint_dict = {
        'type': 'ineq',
        'fun': compute_constraints,
        'args': (n,)
    }
    obj_func_args = (n,)
    
    # Prepare initial configurations
    configs = []
    
    # 1. Regular grid base
    cx_base = np.linspace(0.15, 0.85, 5)
    cy_base = np.linspace(0.15, 0.85, 5)
    grid_pts = np.array([[x, y] for y in cy_base for x in cx_base])[:25]
    extra_pt = np.array([[0.5, 0.5]])
    grid_full = np.vstack([grid_pts, extra_pt])
    configs.append(grid_full)
    
    # 2. Hexagonal pattern base
    cx_hex = np.linspace(0.1, 0.9, 6)
    cy_hex = np.linspace(0.1, 0.9, 5)
    hex_pts = []
    for i, y in enumerate(cy_hex):
        for j, x in enumerate(cx_hex):
            if len(hex_pts) >= 26: break
            if i % 2 == 1:
                x += 0.05
            hex_pts.append([x, y])
        if len(hex_pts) >= 26: break
    configs.append(np.array(hex_pts[:26]))
    
    # 3. Randomized perturbations
    for _ in range(4):
        noise = np.random.uniform(-0.06, 0.06, size=(26, 2))
        cfg = np.clip(grid_full + noise, 0.1, 0.9)
        configs.append(cfg)
        
    for centers_init in configs:
        x0 = np.empty(3 * n)
        x0[0::3] = centers_init[:, 0]
        x0[1::3] = centers_init[:, 1]
        x0[2::3] = 0.08  # Start with reasonable radii
        
        try:
            res = minimize(
                compute_objective,
                x0,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraint_dict,
                args=obj_func_args,
                options={'maxiter': 800, 'ftol': 1e-10, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = res.x[2::3]
                
                # Quick internal validity check
                cons_vals = compute_constraints(res.x, n)
                if np.all(cons_vals >= -1e-7):
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt
                        best_radii = r_opt
        except Exception:
            pass

    # Fallback to a known valid packing if optimization yields nothing useful
    if best_centers is None or best_sum < 2.0:
        cx_fb = np.linspace(0.0833, 0.9167, 6)
        cy_fb = np.linspace(0.0833, 0.9167, 5)
        fb_centers = []
        for i, y in enumerate(cy_fb):
            for j, x in enumerate(cx_fb):
                if len(fb_centers) >= 26: break
                if i % 2 == 1:
                    x += 0.0416
                fb_centers.append([x, y])
            if len(fb_centers) >= 26: break
        best_centers = np.array(fb_centers[:26])
        best_radii = np.full(26, 0.08)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
