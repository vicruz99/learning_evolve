# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000053 (state 2e035c71) state=9320a407 sum of radii=2.623068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

# Precompute pair indices for vectorized constraints
PAIR_I = np.array([i for i in range(N) for j in range(i + 1, N)])
PAIR_J = np.array([j for i in range(N) for j in range(i + 1, N)])

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Vectorized inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap (squared distance formulation for better conditioning)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist_sq = dx**2 + dy**2
    r_sum = r[PAIR_I] + r[PAIR_J]
    
    return np.concatenate([c, dist_sq - r_sum**2])

def get_feasible_radii(cx, cy):
    """Compute strictly feasible initial radii for a given set of centers."""
    wall_dists = np.minimum(np.minimum(cx, 1.0 - cx), np.minimum(cy, 1.0 - cy))
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    pair_dists = np.min(dists / 2.0, axis=1)
    return np.minimum(wall_dists, pair_dists)

def generate_start(seed, strategy):
    """Generate a diverse, strictly feasible initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    
    if strategy == 0:  # Hexagonal lattice with random rotation
        r0 = 0.09 + np.random.uniform(-0.01, 0.01)
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 10:
            x_start = r0 if row % 2 == 0 else 2 * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 10:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        centers = np.array(pts[:N])
        # Random rotation to break symmetry
        angle = np.random.uniform(-0.2, 0.2)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        centers = (rot @ (centers - 0.5).T).T + 0.5
        centers = np.clip(centers + np.random.uniform(-0.02, 0.02, centers.shape), 0.05, 0.95)
        
    elif strategy == 1:  # Random uniform scatter
        centers = np.random.uniform(0.1, 0.9, (N, 2))
        
    elif strategy == 2:  # Structured row-count variant (6-5-6-5-4)
        pts = []
        counts = [6, 5, 6, 5, 4]
        y = 0.1
        for row_idx, cnt in enumerate(counts):
            x_start = 0.1 + (0.02 if row_idx % 2 == 1 else 0.0)
            step = 0.8 / (cnt - 1 if cnt > 1 else 1)
            for c in range(cnt):
                pts.append([x_start + c * step, y])
            y += 0.2
        centers = np.array(pts[:N])
        centers = np.clip(centers + np.random.uniform(-0.02, 0.02, centers.shape), 0.05, 0.95)
        
    # Initialize radii to 85% of theoretical maximum to guarantee feasibility
    r_init = get_feasible_radii(centers[:, 0], centers[:, 1]) * 0.85
    return np.concatenate([centers[:, 0], centers[:, 1], r_init])

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Phase 1: Multi-start optimization from diverse layouts
    for i in range(50):
        strategy = i % 3
        seed = i // 3
        v0 = generate_start(seed, strategy)
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-10, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Accept if sufficiently feasible
                if np.min(constraints(res.x)) >= -1e-5:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = generate_start(0, 0)
        
    # Phase 2: Perturbation refinement to escape local minima
    current_v = best_v.copy()
    current_sum = best_sum
    
    for step in range(20):
        pert = current_v.copy()
        # Perturb centers
        pert[:2*N] += np.random.uniform(-0.006, 0.006, 2*N)
        pert[:2*N] = np.clip(pert[:2*N], 0.02, 0.98)
        
        # Recompute feasible radii to ensure valid start for optimizer
        r_feas = get_feasible_radii(pert[:N], pert[N:2*N])
        pert[2*N:] = r_feas * 0.95
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
            if -res.fun > current_sum:
                if np.min(constraints(res.x)) >= -1e-5:
                    current_sum = -res.fun
                    current_v = res.x.copy()
                    best_sum = current_sum
                    best_v = current_v.copy()
        except Exception:
            pass

    # Phase 3: Extract and strictly enforce constraints for validator compliance
    cx = best_v[:N].copy()
    cy = best_v[N:2*N].copy()
    cr = best_v[2*N:].copy()
    
    # Strict boundary enforcement
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # Strict non-overlap enforcement with iterative shrinkage
    for _ in range(5):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < cr[i] + cr[j] - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-7
                    cr[i] -= shrink
                    cr[j] -= shrink
        cr = np.maximum(cr, 0.0)
        
    # Final boundary clamp to prevent any floating-point boundary violations
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
