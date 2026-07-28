# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=8609ace4 sum of radii=2.630698 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_arr, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_arr[2*n:])

def constraints(vars_arr, n):
    """Compute all inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    c_bound = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Pairwise non-overlap constraints: squared distance >= squared sum of radii
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    dist_sq = dx**2 + dy**2
    r_sum_sq = dr**2
    
    iu, ju = np.triu_indices(n, k=1)
    c_pairwise = dist_sq[iu, ju] - r_sum_sq[iu, ju]
    
    return np.concatenate([c_bound, c_pairwise])

def generate_hex_layout(row_counts, r0, n):
    """Generate initial positions on a hexagonal lattice with specified row counts."""
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
    # Pad with center points if layout yields fewer than n
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-7, 0.5)] * n
    
    # Diverse row configurations that sum to >= 26, tailored for hexagonal packing
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4],
        [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 5, 5, 5, 5, 1], [4, 5, 6, 6, 5], [5, 7, 5, 6, 3]
    ]
    
    inits = []
    rng = np.random.default_rng(42)
    
    for rp in row_patterns:
        base = generate_hex_layout(rp, r0=0.088, n=n)
        inits.append(base)
        # Add controlled perturbations to escape symmetry traps
        for _ in range(5):
            pert = base + rng.uniform(-0.012, 0.012, base.shape)
            inits.append(np.clip(pert, 0.05, 0.95))
            
    best_sum = -1.0
    best_c = None
    best_r = None
    
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    # Phase 1: Initial Optimization
    for cfg in inits:
        v0 = np.zeros(3*n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = 0.065  # Small initial radius guarantees feasibility
        
        try:
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=cons_dict,
                args=(n,),
                options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
            )
            
            cx = res.x[:n]
            cy = res.x[n:2*n]
            rs = res.x[2*n:]
            
            if np.any(rs < 1e-7): continue
            
            # Strict validity check matching grader tolerance
            if np.any(cx - rs < -1e-9) or np.any(cx + rs > 1.0 + 1e-9) or \
               np.any(cy - rs < -1e-9) or np.any(cy + rs > 1.0 + 1e-9):
                continue
                
            dx = cx[:, None] - cx[None, :]
            dy = cy[:, None] - cy[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            rs_sum = rs[:, None] + rs[None, :]
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if np.any(dist[mask] < rs_sum[mask] - 1e-9):
                continue
                
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_c = np.column_stack((cx, cy))
                best_r = rs.copy()
        except Exception:
            continue
            
    # Phase 2: Refinement to escape local minima
    if best_c is not None:
        for _ in range(6):
            v_curr = np.concatenate([best_c.flatten(), best_r])
            scale_pert = 0.004 * (1.0 + rng.random())
            pert = rng.normal(0, scale_pert, v_curr.shape)
            v_pert = v_curr + pert
            
            # Enforce bounds on perturbed start
            v_pert[:n] = np.clip(v_pert[:n], 0.0, 1.0)
            v_pert[n:2*n] = np.clip(v_pert[n:2*n], 0.0, 1.0)
            v_pert[2*n:] = np.clip(v_pert[2*n:], 1e-7, 0.5)
            
            try:
                res = minimize(
                    objective, 
                    v_pert, 
                    method='SLSQP', 
                    bounds=bounds,
                    constraints=cons_dict,
                    args=(n,),
                    options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False}
                )
                
                cx = res.x[:n]
                cy = res.x[n:2*n]
                rs = res.x[2*n:]
                
                if np.any(rs < 1e-7): continue
                if np.any(cx - rs < -1e-9) or np.any(cx + rs > 1.0 + 1e-9) or \
                   np.any(cy - rs < -1e-9) or np.any(cy + rs > 1.0 + 1e-9):
                    continue
                    
                dx = cx[:, None] - cx[None, :]
                dy = cy[:, None] - cy[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                rs_sum = rs[:, None] + rs[None, :]
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.any(dist[mask] < rs_sum[mask] - 1e-9):
                    continue
                    
                s = np.sum(rs)
                if s > best_sum:
                    best_sum = s
                    best_c = np.column_stack((cx, cy))
                    best_r = rs.copy()
            except Exception:
                continue

    # Fallback if optimization unexpectedly fails
    if best_c is None:
        best_c = inits[0]
        best_r = np.full(n, 0.08)
        best_sum = np.sum(best_r)
        
    # Phase 3: Safety scaling to guarantee strict numerical validity
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
                
    best_r *= scale * 0.999999
    best_sum = np.sum(best_r)
    
    return best_c, best_r, float(best_sum)
