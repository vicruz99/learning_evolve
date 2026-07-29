# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000167 (state d81766f0) state=0ce7a01b sum of radii=2.629515 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap distances."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    c = np.empty(4*N + len(PAIR_I))
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    return c

def get_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.minimum(np.minimum(centers[:,0], 1.0-centers[:,0]), 
                   np.minimum(centers[:,1], 1.0-centers[:,1]))
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    r = np.minimum(r, np.min(dists, axis=1) / 2.0)
    return np.clip(r * 0.90, 0.01, 0.25)

def generate_hex_config(r0, shift_x, shift_y, angle):
    """Generate a hexagonal lattice configuration with specific parameters."""
    pts = []
    y = r0 + shift_y
    row = 0
    # Row distribution tailored for N=26
    row_counts = [5, 6, 5, 6, 4]
    while len(pts) < N:
        cnt = row_counts[row % len(row_counts)]
        x_start = r0 + shift_x + (row % 2) * r0
        x = x_start
        for _ in range(cnt):
            if len(pts) >= N:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
        row += 1
        
    pts = np.array(pts[:N])
    if abs(angle) > 1e-6:
        c_v, s_v = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_v, -s_v], [s_v, c_v]]) + 0.5
    return pts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.25)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = []
    
    # 1. Hexagonal lattice variations covering optimal basin
    for r0 in [0.095, 0.100, 0.105]:
        for ang in np.linspace(-0.15, 0.15, 7):
            for sx in np.linspace(-0.02, 0.02, 4):
                for sy in np.linspace(-0.02, 0.02, 4):
                    p = generate_hex_config(r0, sx, sy, ang)
                    if np.all((p[:,0]>0.02) & (p[:,0]<0.98) & (p[:,1]>0.02) & (p[:,1]<0.98)):
                        inits.append(p.copy())
                        
    # 2. Force-relaxed random starts for asymmetry/boundary effects
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(150):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 0.25 and d > 1e-4:
                        f = (0.25 - d) / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.04
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    # Phase 1: Multi-start optimization
    for c_init in inits:
        r_init = get_feasible_radii(c_init)
        v0 = np.concatenate([c_init[:,0], c_init[:,1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 25000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-8:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        return np.zeros((N,2)), np.zeros(N), 0.0
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    curr_v = best_v.copy()
    for step in range(50):
        np.random.seed(step + 200)
        v_p = curr_v.copy()
        
        # Decaying noise scale for annealing-like behavior
        noise_scale = 0.003 * (1.0 - step / 50.0)
        v_p[:2*N] += np.random.normal(0, noise_scale, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        
        # Slight radius inflation to encourage expansion, then shrink to feasible
        v_p[2*N:] *= 1.005
        c_pts = v_p[:2*N].reshape(N, 2)
        v_p[2*N:] = get_feasible_radii(c_pts) * 0.95
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-8:
                    best_sum = s
                    best_v = res.x.copy()
                    curr_v = best_v.copy()
        except Exception:
            pass
            
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0-centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0-centers[:,1]))
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    # Validator allows dist >= r1 + r2 - 1e-12. We use 1e-13 buffer to be safe.
    for _ in range(30):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d + 1e-13:
                    shrink = (radii[i] + radii[j] - d - 1e-13) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
