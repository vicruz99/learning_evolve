# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000053 (state 2e035c71) state=4eb2e47c sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, differential_evolution, minimize

N = 26
PAIRS = np.triu_indices(N, k=1)

def compute_lp_sum(centers):
    """Solves LP to find maximum sum of radii for fixed centers."""
    x, y = centers[:, 0], centers[:, 1]
    # Boundary constraints: r_i <= distance to nearest wall
    b = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    b = np.maximum(b, 1e-9)
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    
    # Construct LP matrices: Maximize sum(r) -> Minimize -sum(r)
    # Constraints: r_i <= b_i  AND  r_i + r_j <= d_ij
    A1 = np.eye(N)
    b1 = b.copy()
    
    m = len(PAIRS[0])
    A2 = np.zeros((m, N))
    A2[np.arange(m), PAIRS[0]] = 1.0
    A2[np.arange(m), PAIRS[1]] = 1.0
    b2 = d[PAIRS]
    
    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([b1, b2])
    
    res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    return -res.fun, -res.x

def objective_func(v):
    """Objective for optimizer: maximize sum of radii -> minimize negative sum."""
    centers = v.reshape(-1, 2)
    val, _ = compute_lp_sum(centers)
    return -val

def run_packing():
    bounds = [(0.02, 0.98)] * (2 * N)
    
    # Generate high-quality initial population
    inits = []
    r_hex = 0.095
    base_pts = []
    y = r_hex
    row = 0
    while len(base_pts) < N + 10:
        x_start = r_hex if row % 2 == 0 else 2 * r_hex
        x = x_start
        while x <= 1.0 - r_hex:
            base_pts.append([x, y])
            x += 2 * r_hex
        y += r_hex * np.sqrt(3)
        row += 1
    base_pts = np.array(base_pts[:N])
    
    np.random.seed(42)
    for _ in range(10):
        pts = base_pts + np.random.uniform(-0.025, 0.025, size=base_pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts.flatten())
        
    # Add grid-based starts for diversity
    grid = np.array([(i*0.18+0.1, j*0.18+0.1) for i in range(5) for j in range(5)] + [(0.5, 0.5)])
    for _ in range(5):
        pts = grid + np.random.uniform(-0.02, 0.02, size=grid.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts.flatten())
        
    inits = np.array(inits)
    
    # Global search over center positions
    try:
        res_de = differential_evolution(objective_func, bounds, 
                                        initial_guess=inits, popsize=15, maxiter=300, 
                                        tol=1e-7, seed=42, workers=1, polish=False)
        best_v = res_de.x
    except Exception:
        best_v = inits[0]
        
    # Local refinement to escape shallow basins
    try:
        res_local = minimize(objective_func, best_v, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res_local.fun < objective_func(best_v):
            best_v = res_local.x
    except Exception:
        pass
        
    centers = best_v.reshape(N, 2)
    _, radii = compute_lp_sum(centers)
    
    # Strict post-processing to guarantee validator compliance
    cr = radii.copy()
    cx, cy = centers[:, 0], centers[:, 1]
    
    # Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    # Enforce non-overlap constraints iteratively with safety margin
    for _ in range(5):
        for i in range(N):
            for j in range(i+1, N):
                dist = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if dist < cr[i] + cr[j] - 1e-9:
                    shrink = (cr[i] + cr[j] - dist) / 2.0 + 1e-7
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    
    cr = np.maximum(cr, 0.0)
    # Final boundary clamp
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    return centers, cr, float(np.sum(cr))
