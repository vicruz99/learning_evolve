# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000069 (state 2c1a60b6) state=5097bdf0 sum of radii=2.316902 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_radii(centers):
    """Computes the maximum valid radius for each circle given fixed centers."""
    x, y = centers[:, 0], centers[:, 1]
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist, np.inf)
    
    # Radius limited by half the distance to the nearest neighbor
    r_pair = 0.5 * np.min(dist, axis=1)
    
    return np.minimum(r_bound, r_pair)

def obj_func(v):
    """Objective: minimize negative sum of radii."""
    c = v.reshape(N, 2)
    c = np.clip(c, 1e-7, 1.0 - 1e-7)
    return -np.sum(compute_radii(c))

def generate_hex_init(seed, density=0.095, rot=0.0):
    """Generates a hexagonal lattice initialization, rotated and scaled."""
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(-5, 10):
        for j in range(-5, 10):
            px = i * density + (j % 2) * 0.5 * density
            py = j * density * np.sqrt(3) / 2.0
            pts.append([px, py])
    pts = np.array(pts[:N])
    
    # Rotation
    c, s = np.cos(rot), np.sin(rot)
    rot_pts = pts @ np.array([[c, -s], [s, c]])
    
    # Center and scale to fit comfortably inside [0,1]
    rot_pts -= rot_pts.mean(axis=0)
    scale = np.max(np.abs(rot_pts))
    if scale > 1e-9:
        rot_pts = rot_pts / scale * 0.45 + 0.5
        
    # Add small deterministic jitter to break symmetry
    rot_pts += rng.normal(0, 0.002, rot_pts.shape)
    return np.clip(rot_pts, 0.02, 0.98).flatten()

def generate_random_init(seed):
    """Generates a dense random initialization."""
    rng = np.random.RandomState(seed)
    c = rng.rand(N, 2) * 0.8 + 0.1
    return c.flatten()

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = [(1e-7, 1.0 - 1e-7)] * (2 * N)
    
    best_val = np.inf
    best_c = None
    
    # ---------------------------------------------------------
    # Phase 1: Multi-Start Powell Optimization
    # ---------------------------------------------------------
    inits = []
    # Diverse hexagonal lattices
    for seed in range(20):
        density = 0.095 + seed * 0.0015
        rot = seed * 0.025 - 0.1
        inits.append(generate_hex_init(seed, density=density, rot=rot))
    # Random dense starts
    for seed in range(15):
        inits.append(generate_random_init(seed))
        
    for v0 in inits:
        try:
            res = minimize(obj_func, v0, method='Powell', bounds=bounds,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'xtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            continue

    if best_c is None:
        best_c = generate_random_init(0).reshape(N, 2)
        
    # ---------------------------------------------------------
    # Phase 2: Iterative Greedy Local Search
    # ---------------------------------------------------------
    c = best_c.copy()
    step = 0.008
    for epoch in range(400):
        improved = False
        for i in range(N):
            current_sum = np.sum(compute_radii(c))
            # Try 8 cardinal/diagonal directions
            directions = [(step, 0), (-step, 0), (0, step), (0, -step),
                          (step, step), (-step, step), (step, -step), (-step, -step)]
            for dx, dy in directions:
                nc = c.copy()
                nc[i] += [dx, dy]
                nc[i] = np.clip(nc[i], 1e-7, 1.0 - 1e-7)
                s = np.sum(compute_radii(nc))
                if s > current_sum + 1e-11:
                    c = nc
                    improved = True
                    break
        if not improved:
            step *= 0.88
        if step < 1e-7:
            break
            
    # ---------------------------------------------------------
    # Phase 3: Final Powell Polish
    # ---------------------------------------------------------
    try:
        res = minimize(obj_func, c.flatten(), method='Powell', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-14, 'xtol': 1e-14})
        if -res.fun > np.sum(compute_radii(c)):
            c = res.x.reshape(N, 2)
    except Exception:
        pass
        
    # ---------------------------------------------------------
    # Phase 4: Exact Radius Computation & Safety Repair
    # ---------------------------------------------------------
    radii = compute_radii(c)
    
    # Strict safety repair to guarantee validator tolerance (1e-12)
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y, r = c[i, 0], c[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1.0 + 1e-12 or y - r < -1e-12 or y + r > 1.0 + 1e-12:
                valid = False
                break
        if not valid:
            break
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii *= 0.99995
        
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    return c, radii, final_sum
