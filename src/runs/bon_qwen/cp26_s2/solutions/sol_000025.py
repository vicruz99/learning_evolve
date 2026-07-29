# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e154041e) state=aa926f1d sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing():
    n_circles = 26
    
    # Helper to unpack variables
    def unpack(vars_arr):
        # vars_arr is flat [x0, y0, r0, x1, y1, r1, ...]
        x = vars_arr[0::3]
        y = vars_arr[1::3]
        r = vars_arr[2::3]
        return x, y, r

    # Objective: Maximize sum of radii => Minimize negative sum
    def objective(vars_arr):
        _, _, r = unpack(vars_arr)
        return -np.sum(r)

    # Constraint function
    def constraint_func(vars_arr):
        x, y, r = unpack(vars_arr)
        cons_vals = []
        
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        # x - r >= 0
        cons_vals.extend(x - r)
        # 1 - x - r >= 0
        cons_vals.extend(1.0 - x - r)
        # y - r >= 0
        cons_vals.extend(y - r)
        # 1 - y - r >= 0
        cons_vals.extend(1.0 - y - r)
        
        # Overlap constraints: dist^2 >= (r_i + r_j)^2
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                rad_sum = r[i] + r[j]
                cons_vals.append(dist_sq - rad_sum**2)
                
        return np.array(cons_vals)

    constraint_dict = {
        'type': 'ineq',
        'fun': constraint_func
    }

    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = tuple([(0, 1), (0, 1), (0, 0.5)] * n_circles)

    # Initialization strategies
    
    def init_hexagonal():
        """Initialize circles in a hexagonal packing pattern."""
        r_init = 0.09 # Initial radius
        vars_arr = np.zeros(78)
        
        temp_centers = []
        y_curr = r_init
        row_idx = 0
        
        while y_curr + r_init <= 1.0 + 1e-9:
            if row_idx % 2 == 0:
                start_x = r_init
            else:
                start_x = 2 * r_init
            
            curr_x = start_x
            while curr_x + r_init <= 1.0 + 1e-9:
                temp_centers.append((curr_x, y_curr))
                curr_x += 2 * r_init
            
            y_curr += r_init * math.sqrt(3)
            row_idx += 1
            
        # Take first 26
        selected = temp_centers[:26]
        
        for i, (cx, cy) in enumerate(selected):
            vars_arr[3*i] = cx
            vars_arr[3*i+1] = cy
            vars_arr[3*i+2] = r_init
            
        return vars_arr

    def init_random(seed):
        """Random initialization with small radii."""
        vars_arr = np.zeros(78)
        rng = np.random.default_rng(seed)
        
        for i in range(n_circles):
            vars_arr[3*i] = rng.uniform(0.1, 0.9)
            vars_arr[3*i+1] = rng.uniform(0.1, 0.9)
            vars_arr[3*i+2] = 0.02 
            
        return vars_arr

    # Optimization
    best_vars = None
    best_score = np.inf 
    
    # Candidates for start
    starts = [
        init_hexagonal(),
        init_random(123),
        init_random(456),
        init_random(789)
    ]

    for x0 in starts:
        try:
            res = opt.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint_dict],
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            score = res.fun
            
            if np.isfinite(score):
                x, y, r = unpack(res.x)
                valid = True
                
                # Boundary
                if np.any(r < 0): valid = False
                if np.any(x < r - 1e-7) or np.any(x > 1 - r + 1e-7): valid = False
                if np.any(y < r - 1e-7) or np.any(y > 1 - r + 1e-7): valid = False
                
                # Overlap
                if valid:
                    for i in range(n_circles):
                        for j in range(i+1, n_circles):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-7:
                                valid = False
                                break
                        if not valid: break
                
                if valid and score < best_score:
                    best_score = score
                    best_vars = res.x.copy()
                    
        except Exception:
            pass

    if best_vars is None:
        best_vars = init_hexagonal()
        
    x, y, r = unpack(best_vars)
    
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers[i] = [x[i], y[i]]
        radii[i] = r[i]
        
    # Safety clamping
    for i in range(n_circles):
        cx, cy = centers[i]
        margin_x = min(cx, 1 - cx)
        margin_y = min(cy, 1 - cy)
        max_r = min(margin_x, margin_y)
        if radii[i] > max_r:
            radii[i] = max_r
            
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
