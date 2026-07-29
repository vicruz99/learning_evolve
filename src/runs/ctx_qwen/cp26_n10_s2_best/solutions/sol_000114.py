# sol_000114 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000083 (state c6ee3a07) state=386437c7 sum of radii=0.716421 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
PAIR_I, PAIR_J = np.triu_indices(N_CIRCLES, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def constraints(v):
    """
    Inequality constraints: boundaries and pairwise non-overlap.
    Returns array where all elements >= 0 indicate feasibility.
    """
    x = v[:N_CIRCLES]
    y = v[N_CIRCLES:2*N_CIRCLES]
    r = v[2*N_CIRCLES:]
    
    m = 4 * N_CIRCLES + len(PAIR_I)
    c = np.empty(m)
    
    # Boundary constraints
    c[:N_CIRCLES] = x - r
    c[N_CIRCLES:2*N_CIRCLES] = 1.0 - x - r
    c[2*N_CIRCLES:3*N_CIRCLES] = y - r
    c[3*N_CIRCLES:4*N_CIRCLES] = 1.0 - y - r
    
    # Pairwise non-overlap constraints
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx**2 + dy**2)
    c[4*N_CIRCLES:] = dist - r[PAIR_I] - r[PAIR_J]
    
    return c

def generate_hex_start(scale, angle, seed):
    """Generates a hexagonal lattice initialization with controlled perturbation."""
    np.random.seed(seed)
    pts = []
    r0 = 0.1
    y = r0
    row = 0
    counts = [5, 6, 5, 6, 4]
    
    for cnt in counts:
        row_width = (cnt - 1) * 2 * r0
        x_start = 0.5 - row_width / 2.0 + (0.5 * 2 * r0 if row % 2 == 1 else 0.0)
        for k in range(cnt):
            pts.append([x_start + k * 2 * r0, y])
        y += r0 * np.sqrt(3)
        row += 1
        
    pts = np.array(pts[:N_CIRCLES])
    
    # Scale to fit within the square
    pts = pts * scale + (1.0 - scale) * 0.5
    
    # Rotate around center
    if angle != 0:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = np.dot(pts - 0.5, rot.T) + 0.5
        
    # Add jitter to break symmetry
    pts += np.random.uniform(-0.01, 0.01, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    v = np.zeros(3 * N_CIRCLES)
    v[:N_CIRCLES] = pts[:, 0]
    v[N_CIRCLES:2*N_CIRCLES] = pts[:, 1]
    v[2*N_CIRCLES:] = 0.04 + np.random.uniform(0, 0.01, N_CIRCLES)
    return v

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    starts = []
    # Hexagonal pattern variants with different scales and rotations
    for scale in [0.85, 0.9, 0.95]:
        for angle in [0.0, 0.05, -0.05, 0.15, -0.15]:
            for seed in range(5):
                starts.append(generate_hex_start(scale, angle, seed))
                
    # Random dense starts for diversity
    for seed in range(10):
        np.random.seed(seed + 100)
        pts = np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))
        v = np.zeros(3 * N_CIRCLES)
        v[:N_CIRCLES] = pts[:, 0]
        v[N_CIRCLES:2*N_CIRCLES] = pts[:, 1]
        v[2*N_CIRCLES:] = np.random.uniform(0.03, 0.06, N_CIRCLES)
        starts.append(v)
        
    # Phase 1: Multi-start optimization
    for v0 in starts:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-5 and -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative refinement with radius scaling to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        np.random.seed(42)
        for step in range(15):
            # Shrink radii to create breathing room
            current_v[2*N_CIRCLES:] *= 0.96
            # Perturb centers
            pert = np.random.uniform(-0.005, 0.005, 2*N_CIRCLES)
            current_v[:2*N_CIRCLES] += pert
            current_v[:2*N_CIRCLES] = np.clip(current_v[:2*N_CIRCLES], 0.02, 0.98)
            
            try:
                res = minimize(objective, current_v, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-5 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_v = res.x.copy()
                    current_v = res.x.copy()
            except Exception:
                continue
                
    # Fallback safety net
    if best_v is None:
        centers = np.random.rand(N_CIRCLES, 2) * 0.8 + 0.1
        radii = np.full(N_CIRCLES, 0.04)
        return centers, radii, float(np.sum(radii))
        
    centers = best_v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_v[2*N_CIRCLES:].copy()
    
    # Strict post-processing to guarantee validator compliance
    for _ in range(30):
        changed = False
        # Enforce boundary constraints
        for i in range(N_CIRCLES):
            mx = min(centers[i,0], 1.0-centers[i,0])
            my = min(centers[i,1], 1.0-centers[i,1])
            if radii[i] > mx: radii[i] = mx; changed = True
            if radii[i] > my: radii[i] = my; changed = True
            
        # Enforce non-overlap constraints strictly
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d:
                    ov = radii[i] + radii[j] - d
                    shr = ov/2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shr)
                    radii[j] = max(0.0, radii[j] - shr)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
