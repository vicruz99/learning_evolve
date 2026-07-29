# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000053 (state 2e035c71) state=c7e336c8 sum of radii=2.630831 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for efficient constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles must be inside [0, 1]x[0, 1]
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[PAIR_I] + r[PAIR_J]
    c = np.concatenate([c, dist - r_sum])
    
    return c

def generate_initial_guess(seed, pattern='hex'):
    """Generate a feasible initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    
    if pattern == 'hex':
        # Hexagonal lattice with varying base radius
        r0 = 0.075 + seed * 0.004
        y = r0
        row = 0
        idx = 0
        while idx < N:
            x_start = r0 if row % 2 == 0 else 2.0 * r0
            x = x_start
            while x <= 1.0 - r0 and idx < N:
                centers[idx] = [x, y]
                x += 2.0 * r0
                idx += 1
            y += r0 * np.sqrt(3.0)
            row += 1
    elif pattern == 'grid':
        # Staggered grid pattern
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.08 + i * 0.16, 0.08 + j * 0.18])
        pts = np.array(pts)
        # Select N points with jitter
        idx = np.random.choice(len(pts), N, replace=False)
        centers = pts[idx]
    else: # random
        centers = np.random.uniform(0.05, 0.95, (N, 2))
        
    # Add controlled jitter to break symmetry
    centers += np.random.uniform(-0.015, 0.015, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    # Initial radii: small enough to guarantee feasibility
    r = np.full(N, 0.035)
    return np.concatenate([centers[:, 0], centers[:, 1], r])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start exploration
    for seed in range(25):
        for pattern in ['hex', 'grid', 'rand']:
            v0 = generate_initial_guess(seed, pattern)
            try:
                res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
                
                curr_sum = -res.fun
                # Accept if better and sufficiently feasible
                cons_vals = constraints(res.x)
                if np.min(cons_vals) >= -1e-5 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
            except Exception:
                continue
                
    # Fallback initialization
    if best_v is None:
        best_v = generate_initial_guess(0, 'hex')
        
    # Phase 2: Inflation & Refinement Loop
    # Deliberately inflate radii and re-optimize to escape local minima
    for step in range(6):
        scale = 1.0 + 0.012 * (1.0 - step * 0.1)
        v_pert = best_v.copy()
        v_pert[2*N:] *= scale
        
        # Slightly perturb centers to help optimizer find new equilibrium
        v_pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            cons_vals = constraints(res.x)
            if np.min(cons_vals) >= -1e-5 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Extract results
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        mr = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        cr[i] = min(cr[i], mr)
        
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(15):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < cr[i] + cr[j] - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-7
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
