# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000043 (state e63f418f) state=a8d74b77 sum of radii=2.626113 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26
# Precompute pairwise indices for overlap constraints
i_idx, j_idx = np.triu_indices(N, k=1)

def objective(p):
    """Objective: minimize negative sum of radii."""
    return -np.sum(p[2::3])

def constraint_func(p):
    """
    Computes all boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0.
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    c = np.empty(4 * N + N * (N - 1) // 2)
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.hypot(dx, dy)
    c[4*N:] = dist - (r[i_idx] + r[j_idx])
    return c

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def generate_hex_init(r0, seed=0):
    """Generates an initial hexagonal packing configuration with optional noise."""
    rng = np.random.default_rng(seed)
    centers = []
    row_y = r0
    row = 0
    while len(centers) < N:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 + 1e-9 and len(centers) < N:
            centers.append([x, row_y])
            x += 2.0 * r0
        row_y += np.sqrt(3) * r0
        row += 1
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    p = np.zeros(N * 3)
    p[0::3] = centers[:, 0]
    p[1::3] = centers[:, 1]
    p[2::3] = r0
    return p

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_p = None
    best_sum = -1.0
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    # Hexagonal variants with different base radii and noise seeds
    for r0 in [0.08, 0.09, 0.10, 0.11]:
        for seed in range(5):
            inits.append(generate_hex_init(r0, seed))
            
    # Perturbed square grid + 1 extra circle
    for seed in range(10):
        rng = np.random.default_rng(seed)
        cx = np.linspace(0.1, 0.9, 5)
        cy = np.linspace(0.1, 0.9, 5)
        grid = np.array(np.meshgrid(cx, cy)).T.reshape(-1, 2)
        grid = grid + rng.normal(0, 0.01, grid.shape)
        extra = rng.uniform(0.2, 0.8, (1, 2))
        centers = np.vstack([grid, extra])
        centers = np.clip(centers, 0.02, 0.98)
        p = np.zeros(N * 3)
        p[0::3] = centers[:, 0]
        p[1::3] = centers[:, 1]
        p[2::3] = 0.09
        inits.append(p)
        
    # Random dense configurations
    for seed in range(20):
        rng = np.random.default_rng(seed)
        centers = rng.uniform(0.1, 0.9, (N, 2))
        p = np.zeros(N * 3)
        p[0::3] = centers[:, 0]
        p[1::3] = centers[:, 1]
        p[2::3] = 0.06
        inits.append(p)
        
    # Phase 2: Multi-start optimization
    for p0 in inits:
        try:
            res = opt.minimize(objective, p0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
            if np.all(constraint_func(res.x) >= -1e-7):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_p = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Iterative Radius Growth to push density limits
    if best_p is not None:
        for step in range(25):
            curr = best_p.copy()
            curr[2::3] *= 1.003  # Gently expand all radii
            try:
                res = opt.minimize(objective, curr, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
                if np.all(constraint_func(res.x) >= -1e-6):
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_p = res.x.copy()
            except Exception:
                pass
                
        # Phase 4: Local perturbation to escape shallow local minima
        for _ in range(30):
            p_trial = best_p.copy()
            p_trial += np.random.normal(0, 0.0005, p_trial.shape)
            p_trial[0::3] = np.clip(p_trial[0::3], 0.001, 0.999)
            p_trial[1::3] = np.clip(p_trial[1::3], 0.001, 0.999)
            p_trial[2::3] = np.clip(p_trial[2::3], 0.001, 0.499)
            try:
                res = opt.minimize(objective, p_trial, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 2000, 'ftol': 1e-14})
                if np.all(constraint_func(res.x) >= -1e-7):
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_p = res.x.copy()
            except Exception:
                pass
                
    # Fallback if all optimizations fail
    if best_p is None:
        best_p = inits[0]
        
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3].copy()
    
    # Phase 5: Strict deterministic repair to guarantee validation compliance
    for _ in range(100):
        changed = False
        # Clamp to boundaries
        for i in range(N):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                        centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > max_r - 1e-12:
                radii[i] = max_r
                changed = True
        # Resolve overlaps proportionally
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) * 0.51
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
