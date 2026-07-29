# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state 7f27ef90) state=f72473f0 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(x):
    """Negative sum of radii to be minimized."""
    return -np.sum(x[2 * N:])

def constraints(x):
    """Boundary and non-overlap inequality constraints (must be >= 0)."""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    c = np.empty(4 * N + N * (N - 1) // 2)
    # Boundary constraints
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    # Pairwise distance constraints
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    dists = np.hypot(dx, dy)
    r_sum = r[I_IDX] + r[J_IDX]
    
    c[4*N:] = dists - r_sum
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_x = None
    
    def try_opt(x0):
        nonlocal best_sum, best_x
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14})
            
            cx = res.x[:N]
            cy = res.x[N:2*N]
            r = res.x[2*N:]
            
            # Strict feasibility check before updating best
            if np.any(r < -1e-8):
                return
            if np.any(cx < r - 1e-8) or np.any(cx + r > 1.0 + 1e-8):
                return
            if np.any(cy < r - 1e-8) or np.any(cy + r > 1.0 + 1e-8):
                return
                
            dx = cx[I_IDX] - cx[J_IDX]
            dy = cy[I_IDX] - cy[J_IDX]
            dists = np.hypot(dx, dy)
            r_sum = r[I_IDX] + r[J_IDX]
            if np.any(dists < r_sum - 1e-8):
                return
                
            val = -res.fun
            if val > best_sum:
                best_sum = val
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 1: Diverse Hexagonal Lattice Initializations
    for seed in range(35):
        rng = np.random.RandomState(seed)
        cx, cy = np.zeros(N), np.zeros(N)
        idx = 0
        spacing = 0.185 + rng.uniform(-0.012, 0.012)
        margin = 0.05
        y = margin
        row = 0
        while idx < N and y < 1.0 - margin:
            x = margin + (row % 2) * spacing / 2.0
            while x < 1.0 - margin and idx < N:
                cx[idx] = x + rng.uniform(-0.006, 0.006)
                cy[idx] = y + rng.uniform(-0.006, 0.006)
                idx += 1
                x += spacing
            y += spacing * math.sqrt(3) / 2.0
            row += 1
            
        r0 = 0.04 + rng.uniform(0.0, 0.005)
        x0 = np.concatenate([cx, cy, np.full(N, r0)])
        try_opt(x0)

    # Phase 2: Random Initializations in Safe Core
    for seed in range(20):
        rng = np.random.RandomState(100 + seed)
        cx = rng.uniform(0.12, 0.88, N)
        cy = rng.uniform(0.12, 0.88, N)
        r0 = 0.035
        x0 = np.concatenate([cx, cy, np.full(N, r0)])
        try_opt(x0)
        
    # Phase 3: Basin Hopping / Local Refinement
    if best_x is not None:
        for it in range(40):
            rng = np.random.RandomState()
            # Decay amplitude to focus search
            amp = 0.0045 * (1.0 - it / 50.0)
            x_pert = best_x + rng.randn(3 * N) * amp
            x_pert = np.clip(x_pert, [b[0] for b in bounds], [b[1] for b in bounds])
            try_opt(x_pert)
            
    # Phase 4: Strict Post-Processing & Validation Projection
    if best_x is not None:
        cx = best_x[:N].copy()
        cy = best_x[N:2*N].copy()
        r = best_x[2*N:].copy()
        
        # Enforce boundary constraints strictly
        for i in range(N):
            max_r = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
            if r[i] > max_r:
                r[i] = max(0.0, max_r - 1e-9)
                
        # Iteratively resolve any remaining pairwise overlaps
        for _ in range(200):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
                    if d < r[i] + r[j] - 1e-12:
                        excess = r[i] + r[j] - d + 1e-11
                        r[i] -= excess / 2.0
                        r[j] -= excess / 2.0
                        changed = True
            if not changed:
                break
                
        r = np.maximum(r, 0.0)
        best_sum = np.sum(r)
        centers = np.column_stack((cx, cy))
        return centers, r, float(best_sum)
        
    # Fallback (should not be reached given the extensive search)
    centers = np.tile([0.5, 0.5], (N, 1))
    radii = np.full(N, 0.02)
    return centers, radii, float(np.sum(radii))
