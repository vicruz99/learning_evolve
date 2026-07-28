# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000010 (state f39c4564) state=d6932792 sum of radii=0.441234 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_and_grad(params, n, mu):
    """
    Computes the objective value (negative sum of radii) and its analytical gradient.
    Includes penalty terms for boundary and pairwise overlap violations.
    """
    cxs = params[0::3]
    cys = params[1::3]
    rs = params[2::3]
    
    val = -np.sum(rs)
    grad = np.zeros_like(params)
    grad[2::3] -= 1.0  # Gradient of -sum(r)
    
    # Boundary penalties: max(0, violation)^2
    # Left: x - r >= 0  => viol = r - x
    viol_x_neg = np.maximum(0.0, rs - cxs)
    val += mu * np.sum(viol_x_neg**2)
    grad[0::3] -= 2.0 * mu * viol_x_neg
    grad[2::3] += 2.0 * mu * viol_x_neg
    
    # Right: 1 - x - r >= 0 => viol = r + x - 1
    viol_x_pos = np.maximum(0.0, rs + cxs - 1.0)
    val += mu * np.sum(viol_x_pos**2)
    grad[0::3] += 2.0 * mu * viol_x_pos
    grad[2::3] += 2.0 * mu * viol_x_pos
    
    # Bottom: y - r >= 0 => viol = r - y
    viol_y_neg = np.maximum(0.0, rs - cys)
    val += mu * np.sum(viol_y_neg**2)
    grad[1::3] -= 2.0 * mu * viol_y_neg
    grad[2::3] += 2.0 * mu * viol_y_neg
    
    # Top: 1 - y - r >= 0 => viol = r + y - 1
    viol_y_pos = np.maximum(0.0, rs + cys - 1.0)
    val += mu * np.sum(viol_y_pos**2)
    grad[1::3] += 2.0 * mu * viol_y_pos
    grad[2::3] += 2.0 * mu * viol_y_pos
    
    # Pairwise penalties: dist(i,j) >= r_i + r_j => viol = r_i + r_j - dist
    for i in range(n):
        xi, yi, ri = cxs[i], cys[i], rs[i]
        for j in range(i + 1, n):
            xj, yj, rj = cxs[j], cys[j], rs[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
                dx, dy = 1e-6, 1e-6
                
            viol = np.maximum(0.0, ri + rj - dist)
            if viol > 1e-9:
                val += mu * viol**2
                fac = 2.0 * mu * viol
                # d(viol)/dx_i = -dx/dist, d(viol)/dx_j = dx/dist
                term_x = -fac * dx / dist
                term_y = -fac * dy / dist
                
                grad[3*i] += term_x
                grad[3*i + 1] += term_y
                grad[3*j] -= term_x
                grad[3*j + 1] -= term_y
                
                # d(viol)/dr_i = 1, d(viol)/dr_j = 1
                grad[3*i + 2] += fac
                grad[3*j + 2] += fac
                
    return val, grad

def run_packing() -> tuple:
    n = 26
    best_sum = 0.0
    best_res = None
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # --- Generate Diverse Initial Configurations ---
    configs = []
    
    # 1. Hexagonal lattice (theoretically densest)
    pts = []
    r_init = 0.095
    row = 0
    y = r_init
    dy = np.sqrt(3) * r_init
    while y + r_init < 1.0:
        x = r_init
        if row % 2 == 1: x += r_init
        while x + r_init < 1.0:
            pts.append([x, y])
            x += 2 * r_init
        y += dy
        row += 1
        if len(pts) > 30: break
        
    if len(pts) >= n:
        configs.append(np.array(pts[:n]))
    else:
        # Fallback to grid if hex generation is short
        gx = np.linspace(0.15, 0.85, 6)
        gy = np.linspace(0.15, 0.85, 5)
        grid = np.array([[x, y] for y in gy for x in gx])
        configs.append(grid[:n])
        
    # 2. Perturbed Hex/Config (breaks symmetry, escapes traps)
    base_cfg = configs[0]
    for _ in range(4):
        cfg = base_cfg.copy()
        cfg += np.random.uniform(-0.03, 0.03, cfg.shape)
        cfg = np.clip(cfg, 0.1, 0.9)
        configs.append(cfg)
        
    # 3. Grid + perturbations
    gx = np.linspace(0.15, 0.85, 5)
    gy = np.linspace(0.15, 0.85, 5)
    grid = np.array([[x, y] for y in gy for x in gx])[:25]
    grid = np.vstack([grid, [0.5, 0.5]])
    configs.append(grid)
    for _ in range(2):
        cfg = grid + np.random.uniform(-0.04, 0.04, grid.shape)
        cfg = np.clip(cfg, 0.1, 0.9)
        configs.append(cfg)
        
    # --- Optimization ---
    mu = 200.0  # Penalty weight
    for cfg in configs:
        x0 = np.empty(3*n)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = 0.085  # Start with reasonable radii
        
        res = minimize(obj_and_grad, x0, jac=True, args=(n, mu), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
                       
        if np.isfinite(res.fun):
            rs = res.x[2::3]
            s = np.sum(rs)
            cxs = res.x[0::3]
            cys = res.x[1::3]
            
            # Quick validity check
            valid = True
            if np.any(rs < 1e-9) or np.any(cxs < 0) or np.any(cxs > 1) or np.any(cys < 0) or np.any(cys > 1):
                valid = False
            if valid:
                dists = np.sqrt(((cxs[:, None] - cxs[None, :])**2 + (cys[:, None] - cys[None, :])**2))
                req = rs[:, None] + rs[None, :]
                np.fill_diagonal(dists, np.inf)
                if np.any(dists < req - 1e-5):
                    valid = False
                    
            if valid and s > best_sum:
                best_sum = s
                best_res = res.x.copy()
                
    # Ultimate fallback if optimization completely fails
    if best_res is None:
        best_res = np.zeros(3*n)
        best_res[0::3] = np.linspace(0.2, 0.8, n)
        best_res[1::3] = np.linspace(0.2, 0.8, n)
        best_res[2::3] = 0.05
        best_sum = 26 * 0.05
        
    centers = np.column_stack((best_res[0::3], best_res[1::3]))
    radii = best_res[2::3]
    
    # --- Final Safety Scaling ---
    # Compute max scaling factor k such that k*radii strictly satisfies all constraints
    k = 1.0
    for i in range(n):
        r = radii[i]
        if r > 1e-9:
            k = min(k, centers[i, 0]/r, (1-centers[i, 0])/r, centers[i,1]/r, (1-centers[i,1])/r)
            
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            r_sum = radii[i] + radii[j]
            if r_sum > 1e-9:
                k = min(k, dist / r_sum)
                
    if k > 0:
        # Apply with tiny margin to pass numerical tolerance in validation
        radii *= k * 0.999999
        best_sum = np.sum(radii)
    else:
        best_sum = 0.0
        
    return centers, radii, float(best_sum)
