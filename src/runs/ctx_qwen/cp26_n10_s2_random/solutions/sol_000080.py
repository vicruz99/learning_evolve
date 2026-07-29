# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000058 (state f7fedeb3) state=4f0eeeb1 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for pairwise constraints to avoid recomputation overhead
PAIR_IDX = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundary and non-overlap (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    
    cons = []
    # Boundary constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    sr = r[:, None] + r[None, :]
    cons.append(dist[PAIR_IDX] - sr[PAIR_IDX])
    
    return np.concatenate(cons)

def get_hex_init(seed, scale=1.0):
    """Generate hexagonal lattice initialization with slight noise."""
    rng = np.random.default_rng(seed)
    centers = []
    r0 = 0.10 * scale
    y = r0
    row = 0
    while len(centers) < N:
        x = r0 + (row % 2) * r0
        while x + r0 <= 1.0 and len(centers) < N:
            centers.append([x, y])
            x += 2 * r0
        y += r0 * np.sqrt(3)
        row += 1
    c = np.array(centers[:N])
    c += rng.normal(0, 0.003, c.shape)
    return c

def run_packing():
    np.random.seed(42)
    best_v = None
    best_val = -np.inf
    
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # ---------------------------------------------------------
    # Phase 1: Diverse Initial Configurations
    # ---------------------------------------------------------
    starts = []
    
    # Hexagonal lattices with varying densities
    for i in range(8):
        c = get_hex_init(i, scale=0.90 + i*0.015)
        starts.append(np.concatenate([c.flatten(), np.full(N, 0.07)]))
        
    # Corner-biased configurations (helps fill corners efficiently)
    for _ in range(5):
        c = np.random.uniform(0.05, 0.95, (N, 2))
        corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        for k, cr in enumerate(corners):
            c[k] = cr
        starts.append(np.concatenate([c.flatten(), np.full(N, 0.07)]))
        
    # Random starts for exploring non-lattice optima
    for _ in range(7):
        c = np.random.uniform(0.08, 0.92, (N, 2))
        starts.append(np.concatenate([c.flatten(), np.full(N, 0.06)]))
        
    # ---------------------------------------------------------
    # Phase 2: Multi-start SLSQP Optimization
    # ---------------------------------------------------------
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            c_vals = constraints(res.x)
            if np.all(c_vals >= -1e-8):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = starts[0]
        
    # ---------------------------------------------------------
    # Phase 3: Iterative Perturbation Refinement
    # ---------------------------------------------------------
    curr_v = best_v.copy()
    for step in range(20):
        pert = curr_v.copy()
        # Perturb centers and radii slightly to escape local minima
        pert[:2*N] += np.random.normal(0, 0.002, 2*N)
        pert[2*N:] += np.random.normal(0, 0.001, N)
        pert = np.clip(pert, 1e-5, 1.0)
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-13})
            c_vals = constraints(res.x)
            if np.all(c_vals >= -1e-8):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x.copy()
                    curr_v = best_v.copy()
        except Exception:
            pass
            
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:]
    
    # ---------------------------------------------------------
    # Phase 4: Deterministic Repair for Strict Validation
    # ---------------------------------------------------------
    for _ in range(30):
        changed = False
        # Resolve overlaps by proportional shrinking
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    ov = radii[i] + radii[j] - d
                    radii[i] -= ov/2 + 1e-9
                    radii[j] -= ov/2 + 1e-9
                    changed = True
        # Clamp to boundaries
        for i in range(N):
            mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mx + 1e-12:
                radii[i] = mx
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
