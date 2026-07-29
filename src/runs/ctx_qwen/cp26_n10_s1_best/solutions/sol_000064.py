# sol_000064 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000045 (state 257d6214) state=f2cbabbe sum of radii=2.625277 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIL_IDX = np.tril_indices(N_CIRCLES, -1)

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def get_constraints(vars):
    """Compute all inequality constraints g(x) >= 0."""
    n = N_CIRCLES
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints (squared distance - sum_radii^2 >= 0)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES

def solve_lp_radii(centers):
    """Given fixed centers, optimally scale radii to maximize sum using LP."""
    n = N_CIRCLES
    c_obj = -np.ones(n)
    
    # Constraints matrix for linprog: A_ub @ r <= b_ub
    # 4 boundary constraints per circle + overlap constraints
    n_bnd = 4 * n
    n_ovl = n * (n - 1) // 2
    A_ub = np.zeros((n_bnd + n_ovl, n))
    b_ub = np.zeros(n_bnd + n_ovl)
    
    idx = 0
    for i in range(n):
        x, y = centers[i]
        # r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if dist > 1e-9:
                A_ub[idx, i] = 1.0
                A_ub[idx, j] = 1.0
                b_ub[idx] = dist
            else:
                # Circles on top of each other, set strict 0 bound
                b_ub[idx] = 0.0
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def generate_force_init(seed, r_target=0.09):
    """Generate initial config using force-directed relaxation."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, 0.02)
    
    for step in range(600):
        forces = np.zeros_like(centers)
        # Repel overlapping circles
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.hypot(dx, dy)
                if dist < radii[i] + radii[j] and dist > 1e-6:
                    push = (radii[i] + radii[j] - dist) * 0.5
                    fx, fy = (push * dx / dist), (push * dy / dist)
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
                    
        # Boundary repulsion
        for i in range(N_CIRCLES):
            if centers[i, 0] < radii[i]: forces[i, 0] += 0.01
            if centers[i, 0] > 1.0 - radii[i]: forces[i, 0] -= 0.01
            if centers[i, 1] < radii[i]: forces[i, 1] += 0.01
            if centers[i, 1] > 1.0 - radii[i]: forces[i, 1] -= 0.01
            
        centers += forces * 0.1
        centers = np.clip(centers, 0.01, 0.99)
        
        # Gradually grow radii towards target
        growth = 0.0005 if step < 200 else 0.0001
        radii = np.minimum(radii + growth, r_target)
        
    return centers, radii

def generate_hex_init(seed, r0=0.09):
    """Generate rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, r0)
    
    idx = 0
    y = r0
    row = 0
    while idx < N_CIRCLES and y + r0 <= 1.0:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while idx < N_CIRCLES and x + r0 <= 1.0:
            centers[idx] = [x, y]
            idx += 1
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    # Rotate slightly to break symmetry
    cx, cy = 0.5, 0.5
    ang = np.random.uniform(-0.2, 0.2)
    c, s = np.cos(ang), np.sin(ang)
    centers -= [cx, cy]
    centers = centers @ np.array([[c, -s], [s, c]])
    centers += [cx, cy]
    centers = np.clip(centers, r0 + 1e-4, 1.0 - r0 - 1e-4)
    
    return centers, radii

def repair_and_validate(centers, radii):
    """Strictly enforce constraints and validate against numerical tolerances."""
    n = N_CIRCLES
    # Boundary clipping
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    # Overlap repair via minimal shrinkage if needed
    for _ in range(50):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    valid = False
                    break
            if not valid: break
            
        if valid: break
        radii *= 0.999
        
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    n = N_CIRCLES
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Phase 1: Diverse Multi-Start Optimization
    inits = []
    # Force-directed starts
    for s in range(10):
        inits.append(generate_force_init(s, r_target=0.10))
    # Hexagonal starts with variations
    for s in range(10):
        inits.append(generate_hex_init(s, r0=0.09))
    # Random perturbed starts
    for s in range(5):
        np.random.seed(s)
        c = np.random.uniform(0.1, 0.9, (n, 2))
        r = np.full(n, 0.07)
        inits.append((c, r))
        
    for centers, radii in inits:
        x0 = np.zeros(3 * n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    # Quick feasibility check
                    c_vals = get_constraints(res.x)
                    if np.min(c_vals) >= -1e-6:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: LP Radius Refinement & Re-optimization
    if best_vars is not None:
        centers_cur = np.column_stack((best_vars[0::3], best_vars[1::3]))
        radii_cur = best_vars[2::3]
        
        # Try LP expansion
        new_radii = solve_lp_radii(centers_cur)
        if new_radii is not None and np.sum(new_radii) > best_sum:
            best_vars[2::3] = new_radii
            best_sum = np.sum(new_radii)
            
        # Re-optimize positions with LP radii as starting point
        x0_lp = best_vars.copy()
        try:
            res = minimize(objective, x0_lp, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun) and -res.fun > best_sum:
                best_vars = res.x.copy()
                best_sum = -res.fun
        except Exception:
            pass

    # Phase 3: Deflation & Basin Hopping to escape local minima
    if best_vars is not None:
        for it in range(20):
            shrink = 0.995 * (0.98**it)
            noise_scale = 0.002 * (0.9**it)
            
            x0 = best_vars.copy()
            x0[2::3] *= shrink # Deflate radii
            x0[0::3] += np.random.normal(0, noise_scale, n)
            x0[1::3] += np.random.normal(0, noise_scale, n)
            
            # Project back to bounds
            for i in range(n):
                r = max(1e-4, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0-r)
                x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0-r)
                x0[3*i+2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_vals = get_constraints(res.x)
                    if np.min(c_vals) >= -1e-6 and -res.fun > best_sum:
                        best_vars = res.x.copy()
                        best_sum = -res.fun
            except Exception:
                pass
                
        # Final LP pass on the best deformed configuration
        centers_cur = np.column_stack((best_vars[0::3], best_vars[1::3]))
        new_radii = solve_lp_radii(centers_cur)
        if new_radii is not None:
            best_vars[2::3] = new_radii
            best_sum = np.sum(new_radii)

    # Fallback (should not be reached)
    if best_vars is None:
        centers_f, radii_f = generate_force_init(0)
        best_vars = np.zeros(3 * n)
        best_vars[0::3] = centers_f[:, 0]
        best_vars[1::3] = centers_f[:, 1]
        best_vars[2::3] = radii_f
        best_sum = np.sum(radii_f)
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3].copy()
    
    # Phase 4: Strict Validation & Repair
    centers, radii = repair_and_validate(centers, radii)
    
    return centers, radii, float(np.sum(radii))
