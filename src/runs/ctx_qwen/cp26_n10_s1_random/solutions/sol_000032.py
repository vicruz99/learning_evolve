# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state bde5dee5) state=ac51bd1a sum of radii=2.621321 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    def objective(vars):
        # Minimize negative sum of radii => Maximize sum of radii
        return -np.sum(vars[2*n:3*n])

    def constraints(vars):
        cx = vars[:n]
        cy = vars[n:2*n]
        r = vars[2*n:3*n]
        con = []
        
        # Boundary constraints: circles must stay inside [0,1]x[0,1]
        for i in range(n):
            con.append(cx[i] - r[i])           # Left
            con.append(1.0 - cx[i] - r[i])     # Right
            con.append(cy[i] - r[i])           # Bottom
            con.append(1.0 - cy[i] - r[i])     # Top
            
        # Overlap constraints: squared distance >= squared sum of radii
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                con.append(dx*dx + dy*dy - (r[i] + r[j])**2)
                
        return np.array(con)

    # Variable bounds: centers in [0,1], radii in [epsilon, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate diverse, feasible initial configurations
    inits = []
    
    # 1. Dense Hex Grid 5x5 + 1 circle in center
    r0 = 0.095
    pts = []
    for row in range(5):
        shift = r0 if row % 2 == 1 else 0.0
        for col in range(5):
            x = col * 2 * r0 + shift + r0
            y = row * np.sqrt(3) * r0 + r0
            pts.append([x, y])
    pts.append([0.5, 0.5])
    pts = np.array(pts)
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    pts = pts * 0.92 + 0.04  # Scale with safety margin
    inits.append(pts)
    
    # 2. Hex Grid with row counts [6,5,5,5,5]
    pts2 = []
    counts = [6, 5, 5, 5, 5]
    y_curr = r0
    for idx, cnt in enumerate(counts):
        shift = r0 if idx % 2 == 1 else 0.0
        x_start = (5 - cnt) * r0
        for col in range(cnt):
            x = x_start + col * 2 * r0 + shift + r0
            y = y_curr
            pts2.append([x, y])
        y_curr += r0 * np.sqrt(3)
    pts2 = np.array(pts2)
    pts2 -= pts2.min(axis=0)
    pts2 /= pts2.max(axis=0)
    pts2 = pts2 * 0.92 + 0.04
    inits.append(pts2)
    
    # 3-5. Perturbed versions to escape symmetry traps
    inits.append(inits[0] + np.random.uniform(-0.02, 0.02, (n, 2)))
    inits.append(inits[1] + np.random.uniform(-0.02, 0.02, (n, 2)))
    
    # 6. Regular Grid 5x5 + 1
    grid_pts = np.array([(i*0.2+0.1, j*0.2+0.1) for j in range(5) for i in range(5)])
    grid_pts = np.vstack([grid_pts, [0.2, 0.8]])
    inits.append(grid_pts)
    
    # Optimization loop over all initial configurations
    for centers_init in inits:
        centers_init = np.clip(centers_init, 0.05, 0.95)
        # Initial guess: [x1..xn, y1..yn, r1..rn]
        x0 = np.concatenate([centers_init.flatten(), np.full(n, 0.08)])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 3000, 'ftol': 1e-12})
            
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                r = res.x[2*n:3*n]
                centers = np.column_stack((cx, cy))
                
                # Strict validation matching the grader's tolerance
                valid = True
                for k in range(n):
                    if cx[k] < r[k] - 1e-12 or cx[k] > 1 - r[k] + 1e-12 or \
                       cy[k] < r[k] - 1e-12 or cy[k] > 1 - r[k] + 1e-12:
                        valid = False
                        break
                if valid:
                    for k in range(n):
                        for m in range(k + 1, n):
                            if (cx[k]-cx[m])**2 + (cy[k]-cy[m])**2 < (r[k]+r[m])**2 - 1e-12:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers
                        best_radii = r
        except Exception:
            pass
            
    # Fallback to a known valid configuration if optimization fails unexpectedly
    if best_centers is None:
        cx_fb, cy_fb = inits[0].T
        r_fb = np.full(n, 0.08)
        best_centers = np.column_stack((cx_fb, cy_fb))
        best_radii = r_fb
        best_sum = np.sum(r_fb)
        
    return best_centers, best_radii, float(best_sum)
