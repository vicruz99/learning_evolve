# sol_000053 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=2e035c71 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Vectorized inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    d_sq = dx**2 + dy**2
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c.append(d_sq[mask] - dr[mask]**2)
    
    return np.concatenate(c)

def run_packing():
    # Bounds: x,y in [0,1], r in [0,0.5]
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N
    
    best_sum = -1.0
    best_v = None
    
    # Base hexagonal lattice configuration
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
    for seed in range(15):
        np.random.seed(seed)
        
        # Perturb lattice to break symmetry and explore basin of attraction
        centers = base_pts + np.random.uniform(-0.04, 0.04, size=base_pts.shape)
        centers = np.clip(centers, 0.01, 0.99)
        
        # Compute strictly feasible initial radii
        r_init = np.full(N, 0.5)
        for i in range(N):
            r_init[i] = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                val = d / 2.0
                if val < r_init[i]: r_init[i] = val
                if val < r_init[j]: r_init[j] = val
        r_init *= 0.7  # Leave slack for optimizer to expand
        
        v0 = np.zeros(3 * N)
        v0[:N] = centers[:, 0]
        v0[N:2*N] = centers[:, 1]
        v0[2*N:] = r_init
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            # Verify feasibility tolerance
            cons_vals = constraints(res.x)
            if np.min(cons_vals) >= -1e-5 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback if optimization fails
    if best_v is None:
        best_v = np.zeros(3*N)
        best_v[2*N:] = 0.01
        
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:]
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    # 2. Enforce non-overlap constraints iteratively
    for _ in range(3):
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if d < cr[i] + cr[j] - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-7
                    cr[i] -= shrink
                    cr[j] -= shrink
        cr = np.maximum(cr, 0.0)
        
    # Final boundary clamp
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
