# sol_000176 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000150 (state 86f9e7dc) state=3c3427aa sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: circles must be inside [0, 1]x[0, 1]
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def make_feasible(v):
    """Iteratively adjust radii to guarantee strict feasibility."""
    cx = v[:N].copy()
    cy = v[N:2*N].copy()
    cr = v[2*N:].copy()
    
    # Enforce boundaries
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    # Enforce pairwise non-overlap
    for _ in range(20):
        dx = cx[PAIR_I] - cx[PAIR_J]
        dy = cy[PAIR_I] - cy[PAIR_J]
        dist = np.hypot(dx, dy)
        overlap = (cr[PAIR_I] + cr[PAIR_J]) - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-7
        cr[PAIR_I] = np.maximum(0.0, cr[PAIR_I] - shrink)
        cr[PAIR_J] = np.maximum(0.0, cr[PAIR_J] - shrink)
        
    return np.concatenate([cx, cy, cr])

def generate_configs():
    """Generates a diverse set of initial configurations."""
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Rotated Hexagonal Lattices
    for r0 in np.linspace(0.08, 0.11, 5):
        for ang in np.linspace(-0.25, 0.25, 7):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 5:
                xs = r0 if row % 2 == 0 else 2 * r0
                x = xs
                while x <= 1.0 - r0 and len(pts) < N + 5:
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3.0) * r0
                row += 1
                
            pts = np.array(pts[:N])
            if ang != 0.0:
                c_val, s_val = np.cos(ang), np.sin(ang)
                pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
                
            pts += rng.uniform(-0.01, 0.01, pts.shape)
            pts = np.clip(pts, 0.02, 0.98)
            
            v = np.concatenate([pts[:, 0], pts[:, 1], np.full(N, 0.04)])
            configs.append(make_feasible(v))
            
    # 2. Force-relaxed random scatter
    for seed in range(25):
        rng_seed = np.random.default_rng(seed + 1000)
        pts = rng_seed.uniform(0.15, 0.85, (N, 2))
        for _ in range(150):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                    if d < 0.22 and d > 1e-5:
                        f = (0.22 - d) * 0.5 / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.02
            pts = np.clip(pts, 0.02, 0.98)
            
        v = np.concatenate([pts[:, 0], pts[:, 1], np.full(N, 0.035)])
        configs.append(make_feasible(v))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    configs = generate_configs()
    
    # Phase 1: Multi-start exploration
    for v0 in configs:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = configs[0]
        best_sum = -objective(best_v)
        
    # Phase 2: Simulated Annealing + Local Optimization refinement
    curr_v = best_v.copy()
    curr_sum = best_sum
    T = 0.006
    for step in range(70):
        # Periodic restart to global best to prevent drifting into poor basins
        if step > 0 and step % 10 == 0:
            curr_v = best_v.copy()
            curr_sum = best_sum
            
        noise = 0.006 * (1.0 - step / 70.0)
        v_p = curr_v.copy()
        v_p[:2*N] += np.random.uniform(-noise, noise, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        v_p[2*N:] *= 0.97  # Shrink to create space for rearrangement
        v_p = make_feasible(v_p)
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
            new_sum = -res.fun
            cv = constraints(res.x)
            if np.min(cv) >= -1e-6:
                if new_sum > curr_sum:
                    curr_sum = new_sum
                    curr_v = res.x.copy()
                else:
                    # Simulated annealing acceptance
                    if np.random.rand() < np.exp((new_sum - curr_sum) / T):
                        curr_sum = new_sum
                        curr_v = res.x.copy()
                        
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_v = res.x.copy()
        except Exception:
            pass
        T *= 0.97  # Cool down
        
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
