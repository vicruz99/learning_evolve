# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=f79bfb57 sum of radii=2.631093 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(vars_flat):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def constraints(vars_flat):
    """
    Inequality constraints: all must be >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    x = vars_flat[0::3]
    y = vars_flat[1::3]
    r = vars_flat[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    b = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: ||c_i - c_j|| >= r_i + r_j
    i, j = np.triu_indices(N, k=1)
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    p = np.sqrt(dx**2 + dy**2) - (r[i] + r[j])
    
    return np.concatenate([b, p])

def generate_hex_config(row_counts, r_init, shift_y=0.0, perturb=0.0):
    """Generates a hexagonal lattice configuration with specified row counts."""
    pts = []
    y = r_init + shift_y
    for ri, cnt in enumerate(row_counts):
        shift_x = r_init if ri % 2 == 1 else 0.0
        x_start = r_init + shift_x
        for _ in range(cnt):
            if len(pts) >= N:
                break
            pts.append([x_start, y])
            x_start += 2.0 * r_init
        y += r_init * np.sqrt(3)
        
    pts = np.array(pts[:N])
    if perturb > 0:
        pts += np.random.uniform(-perturb, perturb, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    np.random.seed(42)
    
    # Bounds: centers in [0, 1], radii in [0.001, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    configs = []
    
    # Hexagonal row patterns that sum to 26
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4],
        [5, 5, 5, 5, 6],
        [6, 6, 5, 5, 4],
        [5, 6, 4, 6, 5],
        [5, 7, 5, 5, 4]
    ]
    
    # Generate diverse initial configurations
    for pat in patterns:
        if sum(pat) != 26: 
            continue
        for r0 in [0.085, 0.095, 0.105]:
            cfg = generate_hex_config(pat, r0)
            configs.append(cfg)
            # Create perturbed variants to escape symmetry traps
            for _ in range(4):
                configs.append(generate_hex_config(pat, r0, perturb=0.015))
                
    # Add fully random configurations for robustness
    for _ in range(8):
        configs.append(np.random.uniform(0.1, 0.9, size=(N, 2)))
        
    # Phase 1: SLSQP Joint Optimization
    for cfg in configs:
        x0 = np.zeros(3 * N)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = 0.05  # Start with a strictly feasible radius
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            # Verify constraints are satisfied within numerical tolerance
            c_vals = constraints(res.x)
            if np.min(c_vals) >= -1e-6:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3])).copy()
                    best_radii = res.x[2::3].copy()
        except Exception:
            continue
            
    # Fallback if optimization fails to find valid configuration
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(N, 0.08)
        best_sum = np.sum(best_radii)
        
    # Phase 2: Linear Programming Refinement for Radii
    # Given fixed centers, maximizing sum(r_i) s.t. r_i + r_j <= dist(i,j) is an LP
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    num_pairs = len(pairs)
    A_ub = np.zeros((num_pairs + 4*N, N))
    b_ub = np.zeros(num_pairs + 4*N)
    
    idx = 0
    for i, j in pairs:
        d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(N):
        x, y = best_centers[i]
        # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        lp_res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success and np.all(lp_res.x >= -1e-9):
            best_radii = np.maximum(lp_res.x, 0.0) * 0.9999999
            best_sum = np.sum(best_radii)
    except Exception:
        # Fallback to SLSQP radii if LP fails
        pass
        
    # Final safety clamp
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, float(best_sum)
