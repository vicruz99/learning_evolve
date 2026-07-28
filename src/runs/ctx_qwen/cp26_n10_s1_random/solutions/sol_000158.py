# sol_000158 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000126 (state 8609ace4) state=c0ae987a sum of radii=2.619994 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective_joint(vars_arr, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_arr[2*n:])

def constraints_joint(vars_arr, n):
    """Compute inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    
    # Boundary constraints
    c_bound = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Pairwise non-overlap constraints (squared distance >= squared sum of radii)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    dist_sq = dx**2 + dy**2
    r_sum_sq = dr**2
    
    iu, ju = np.triu_indices(n, k=1)
    c_pairwise = dist_sq[iu, ju] - r_sum_sq[iu, ju]
    
    return np.concatenate([c_bound, c_pairwise])

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= dist_to_wall
    for i in range(n):
        x, y = centers[i]
        limits = [x, 1.0 - x, y, 1.0 - y]
        for lim in limits:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
            
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x
    except Exception:
        pass
    return None

def generate_hex_layout(row_counts, r0, n):
    """Generate initial positions on a hexagonal lattice."""
    pts = []
    y = r0
    for i, cnt in enumerate(row_counts):
        shift = r0 if i % 2 == 1 else 0.0
        row_width = (cnt - 1) * 2 * r0
        x_start = 0.5 - row_width / 2.0 + shift
        for k in range(cnt):
            if len(pts) >= n: break
            x = x_start + k * 2 * r0
            pts.append([x, y])
        y += np.sqrt(3) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-7, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints_joint, 'args': (n,)}
    
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4],
        [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 5, 5, 5, 5, 1], [4, 5, 6, 6, 5], [5, 7, 5, 6, 3]
    ]
    
    inits = []
    rng = np.random.default_rng(42)
    
    for rp in row_patterns:
        base = generate_hex_layout(rp, r0=0.09, n=n)
        inits.append(base)
        for _ in range(3):
            pert = base + rng.uniform(-0.015, 0.015, base.shape)
            inits.append(np.clip(pert, 0.05, 0.95))
            
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # Phase 1: Global search over diverse configurations
    for cfg in inits:
        v0 = np.zeros(3*n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = 0.085
        
        try:
            res = minimize(
                objective_joint, v0, method='SLSQP', bounds=bounds,
                constraints=cons_dict, args=(n,),
                options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
            )
            
            if not res.success: continue
            
            cx = res.x[:n]
            cy = res.x[n:2*n]
            rs = res.x[2*n:]
            
            if np.any(rs < 1e-7): continue
            if np.any(cx < rs - 1e-9) or np.any(cx > 1.0 - rs + 1e-9) or \
               np.any(cy < rs - 1e-9) or np.any(cy > 1.0 - rs + 1e-9):
               continue
               
            centers_temp = np.column_stack((cx, cy))
            s = np.sum(rs)
            
            # LP refinement for optimal radius assignment
            lp_r = solve_radii_lp(centers_temp)
            if lp_r is not None:
                s_lp = np.sum(lp_r)
                if s_lp > s:
                    s = s_lp
                    rs = lp_r
                    
            if s > best_sum:
                best_sum = s
                best_c = centers_temp.copy()
                best_r = rs.copy()
                
        except Exception:
            continue
            
    # Phase 2: Local perturbation to escape local minima
    if best_c is not None:
        for _ in range(5):
            v_curr = np.concatenate([best_c.flatten(), best_r])
            pert = rng.normal(0, 0.002, v_curr.shape)
            v_pert = v_curr + pert
            v_pert[:n] = np.clip(v_pert[:n], 0.0, 1.0)
            v_pert[n:2*n] = np.clip(v_pert[n:2*n], 0.0, 1.0)
            v_pert[2*n:] = np.clip(v_pert[2*n:], 1e-7, 0.5)
            
            try:
                res2 = minimize(
                    objective_joint, v_pert, method='SLSQP', bounds=bounds,
                    constraints=cons_dict, args=(n,),
                    options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
                )
                if res2.success:
                    cx2 = res2.x[:n]
                    cy2 = res2.x[n:2*n]
                    rs2 = res2.x[2*n:]
                    if np.any(rs2 < 1e-7): continue
                    if np.any(cx2 < rs2 - 1e-9) or np.any(cx2 > 1.0 - rs2 + 1e-9) or \
                       np.any(cy2 < rs2 - 1e-9) or np.any(cy2 > 1.0 - rs2 + 1e-9):
                       continue
                        
                    centers2 = np.column_stack((cx2, cy2))
                    s2 = np.sum(rs2)
                    lp_r2 = solve_radii_lp(centers2)
                    if lp_r2 is not None:
                        s2 = np.sum(lp_r2)
                        rs2 = lp_r2
                        
                    if s2 > best_sum:
                        best_sum = s2
                        best_c = centers2.copy()
                        best_r = rs2.copy()
            except Exception:
                continue

    # Fallback
    if best_c is None:
        best_c = inits[0]
        best_r = np.full(n, 0.08)
        best_sum = np.sum(best_r)
        
    # Phase 3: Strict safety scaling for numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_c[i] - best_c[j])
            rs_sum = best_r[i] + best_r[j]
            if rs_sum > 1e-12:
                scale = min(scale, d / rs_sum)
                
    best_r *= max(scale * 0.9999995, 0.0)
    best_sum = np.sum(best_r)
    
    return best_c, best_r, float(best_sum)
