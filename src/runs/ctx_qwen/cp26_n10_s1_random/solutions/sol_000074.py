# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 81a0d5f4) state=1a4858e6 sum of radii=0.378332 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_eq_constraints(vars):
    """
    Computes inequality constraints for equal-radius packing.
    Constraints must be >= 0.
    vars: [x0, y0, x1, y1, ..., x25, y25, t]
    """
    n = 26
    xs = vars[0::2]
    ys = vars[1::2]
    t = vars[-1]
    
    c = np.empty(4*n + n*(n-1)//2)
    # Boundary constraints: dist to wall >= t
    c[:n] = xs - t
    c[n:2*n] = 1.0 - xs - t
    c[2*n:3*n] = ys - t
    c[3*n:4*n] = 1.0 - ys - t
    
    # Pairwise constraints: dist^2 >= 4*t^2
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c[4*n + idx] = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2 - 4.0 * t * t
            idx += 1
    return c

def get_uneq_constraints(vars):
    """
    Computes inequality constraints for unequal-radius packing.
    Constraints must be >= 0.
    vars: [x0, y0, r0, x1, y1, r1, ...]
    """
    n = 26
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    
    c = np.empty(4*n + n*(n-1)//2)
    # Boundary constraints
    c[:n] = xs - rs
    c[n:2*n] = 1.0 - xs - rs
    c[2*n:3*n] = ys - rs
    c[3*n:4*n] = 1.0 - ys - rs
    
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c[4*n + idx] = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2 - (rs[i] + rs[j])**2
            idx += 1
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_result = None
    
    np.random.seed(42)
    
    # --- Phase 1: Optimize Equal Radius ---
    # Generate diverse hexagonal initial configurations
    init_configs = []
    for _ in range(6):
        pts = []
        r_init = 0.09
        y = r_init
        row = 0
        while len(pts) < n:
            shift = r_init if row % 2 == 1 else 0.0
            x = r_init + shift
            while x <= 1.0 - r_init and len(pts) < n:
                pts.append([x, y])
                x += 2.0 * r_init
            y += np.sqrt(3) * r_init
            row += 1
        pts = np.array(pts[:n])
        # Small random perturbation to break symmetry and explore landscape
        pts += np.random.normal(0, 0.004, pts.shape)
        pts = np.clip(pts, 0.06, 0.94)
        # Flatten to [x0, y0, ..., x25, y25, t]
        x0_eq = np.concatenate([pts.flatten(), [0.085]])
        init_configs.append(x0_eq)
        
    eq_bounds = [(0.0, 1.0)] * (2 * n) + [(0.06, 0.13)]
    cons_eq = {'type': 'ineq', 'fun': get_eq_constraints}
    
    best_t = 0.0
    best_eq_centers = None
    
    for x0 in init_configs:
        try:
            res = minimize(lambda v: -v[-1], x0, method='SLSQP', bounds=eq_bounds,
                           constraints=cons_eq, options={'maxiter': 4000, 'ftol': 1e-14})
            if res.x[-1] > best_t:
                best_t = res.x[-1]
                best_eq_centers = res.x[:2 * n].reshape(n, 2)
        except Exception:
            continue
            
    if best_eq_centers is None:
        # Fallback valid start
        best_eq_centers = np.array([[0.5, 0.5]] * n) + np.random.uniform(-0.2, 0.2, (n, 2))
        best_eq_centers = np.clip(best_eq_centers, 0.1, 0.9)
        best_t = 0.085
        
    # --- Phase 2: Optimize Unequal Radii ---
    # Initialize from Phase 1 optimal centers, with slightly perturbed radii
    x0_uneq = np.zeros(3 * n)
    x0_uneq[0::3] = best_eq_centers[:, 0]
    x0_uneq[1::3] = best_eq_centers[:, 1]
    # Perturb radii to encourage exploration of unequal configurations if beneficial
    x0_uneq[2::3] = best_t * (1.0 + np.random.uniform(-0.02, 0.02, n))
    
    uneq_bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons_uneq = {'type': 'ineq', 'fun': get_uneq_constraints}
    
    try:
        res_uneq = minimize(lambda v: -np.sum(v[2::3]), x0_uneq, method='SLSQP', bounds=uneq_bounds,
                            constraints=cons_uneq, options={'maxiter': 6000, 'ftol': 1e-14})
        
        fc = res_uneq.x[:2 * n].reshape(n, 2)
        fr = res_uneq.x[2 * n:]
        
        # Strict safety scaling to guarantee validation tolerance
        scale = 1.0
        for i in range(n):
            x, y, r = fc[i, 0], fc[i, 1], fr[i]
            if r < 1e-12: continue
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(fc[i] - fc[j])
                rs = fr[i] + fr[j]
                if rs < 1e-12: continue
                scale = min(scale, d / rs)
                
        # Apply shrink with high precision margin
        fr *= max(scale * 0.9999995, 1e-9)
        s = np.sum(fr)
        
        # Final validity check
        valid = True
        for i in range(n):
            x, y, r = fc[i, 0], fc[i, 1], fr[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                valid = False; break
            for j in range(i + 1, n):
                if np.linalg.norm(fc[i] - fc[j]) < fr[i] + fr[j] - 1e-12:
                    valid = False; break
            if not valid: break
            
        if valid and s > best_sum:
            best_sum = s
            best_result = (fc.copy(), fr.copy(), s)
            
    except Exception:
        pass
        
    # Guaranteed fallback if optimization fails unexpectedly
    if best_result is None:
        r_fb = 0.095
        fb_c = np.array([(i * 2 * r_fb + r_fb, j * 2 * r_fb + r_fb) 
                         for j in range(5) for i in range(5)] + [[0.55, 0.55]])
        fb_r = np.full(26, r_fb)
        fb_r[-1] = 0.05
        best_result = (fb_c, fb_r, np.sum(fb_r))
        
    return best_result
