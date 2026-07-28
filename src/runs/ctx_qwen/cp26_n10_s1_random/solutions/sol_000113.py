# sol_000113 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=b0b34868 sum of radii=2.487260 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(vars_flat):
    """Objective: maximize t => minimize -t"""
    return -vars_flat[-1]

def constraints(vars_flat, N):
    """Compute inequality constraints: returns array where each element must be >= 0."""
    xs = vars_flat[0::2]
    ys = vars_flat[1::2]
    t = vars_flat[-1]
    
    c = []
    # Boundary constraints: t <= x <= 1-t and t <= y <= 1-t
    c.extend(xs - t)
    c.extend(1.0 - xs - t)
    c.extend(ys - t)
    c.extend(1.0 - ys - t)
    
    # Pairwise non-overlap constraints: dist^2 >= 4*t^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx**2 + dy**2
    c.extend(dist_sq - 4.0 * t * t)
    
    return np.array(c)

def get_hex_configs(N):
    """Generate multiple high-quality hexagonal initial configurations."""
    configs = []
    # Various row distributions that sum to >= N, tailored for hex packing
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4],
        [5, 6, 4, 6, 5],
        [5, 5, 6, 5, 5]
    ]
    
    for pat in patterns:
        pts = []
        r = 0.101
        h = r * np.sqrt(3)
        total_h = (len(pat)-1)*h + 2*r
        y_start = (1.0 - total_h)/2.0 + r
        
        for ri, cnt in enumerate(pat):
            y_curr = y_start + ri * h
            shift = r if ri % 2 == 1 else 0.0
            w = (cnt-1)*2*r
            x_start = (1.0 - w)/2.0 + shift
            for k in range(cnt):
                pts.append([x_start + k*2*r, y_curr])
        
        if len(pts) >= N:
            configs.append(np.array(pts[:N]))
            
    # Add randomized perturbations to escape shallow local minima
    np.random.seed(42)
    for _ in range(6):
        base = configs[0]
        pert = base + np.random.uniform(-0.025, 0.025, base.shape)
        configs.append(np.clip(pert, 0.05, 0.95))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_t = 0.0
    best_centers = None
    
    configs = get_hex_configs(N)
    
    # Phase 1: Optimize centers to maximize minimum clearance (equal radius t)
    for cfg in configs:
        # Flatten centers and append initial t guess
        x0 = np.concatenate([cfg.flatten(), [0.095]])
        bounds = [(0.0, 1.0)] * (2*N) + [(0.05, 0.15)]
        cons = {'type': 'ineq', 'fun': constraints, 'args': (N,)}
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-11})
            
            # Check if we found a better valid configuration
            if res.x[-1] > best_t:
                # Verify constraints are satisfied within tolerance
                c_vals = constraints(res.x, N)
                if np.min(c_vals) >= -1e-7:
                    best_t = res.x[-1]
                    best_centers = res.x[:2*N].reshape(N, 2)
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = configs[0]
        best_t = 0.09
        
    # Phase 2: Fix centers and solve LP to maximize sum of (possibly unequal) radii
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    num_pairs = len(pairs)
    
    # A_ub * r <= b_ub
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
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        # Maximize sum(r) => minimize -sum(r)
        lp_res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success and np.isfinite(lp_res.fun):
            radii = lp_res.x * 0.9999999  # Tiny buffer for strict 1e-12 validator tolerance
            return best_centers, radii, float(np.sum(radii))
    except Exception:
        pass
        
    # Final fallback
    radii = np.full(N, best_t * 0.99999)
    return best_centers, radii, float(np.sum(radii))
