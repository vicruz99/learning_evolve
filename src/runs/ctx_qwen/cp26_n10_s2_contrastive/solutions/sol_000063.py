# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000038 (state cf517c54) state=de443115 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    def objective(x):
        """Minimize negative sum of radii."""
        return -np.sum(x[2::3])
        
    def constraints(x):
        """Inequality constraints: boundary clearance and pairwise non-overlap."""
        cx = x[0::3]
        cy = x[1::3]
        r = x[2::3]
        
        c = np.empty(4 * n + n_pairs)
        c[:n] = cx - r
        c[n:2*n] = 1.0 - cx - r
        c[2*n:3*n] = cy - r
        c[3*n:4*n] = 1.0 - cy - r
        
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        c[4*n:] = dist - r[i_idx] - r[j_idx]
        return c

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_x = None
    
    def run_opt(x0, iters=5000):
        nonlocal best_sum, best_x
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': iters, 'ftol': 1e-14, 'disp': False})
            if res.success and -res.fun > best_sum:
                best_sum = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Generate diverse initial configurations with symmetry breaking
    configs = []
    rng_init = np.random.RandomState(123)
    
    # 1. Hexagonal lattices with varying spacing
    for s in np.linspace(0.17, 0.215, 8):
        centers = []
        y = s / 2
        row = 0
        while len(centers) < n:
            x = s/2 + (row % 2) * s/2
            while x < 1.0 - s/2 and len(centers) < n:
                centers.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        centers = np.array(centers[:n])
        # Add symmetry-breaking noise
        centers += rng_init.normal(0, 0.008, centers.shape)
        centers = np.clip(centers, 0.06, 0.94)
        r = np.full(n, s * 0.42) + rng_init.normal(0, 0.005, n)
        r = np.clip(r, 0.03, 0.45)
        configs.append(np.column_stack([centers, r]).flatten())
        
    # 2. Square grids
    for s in np.linspace(0.18, 0.22, 6):
        centers = []
        y = s/2
        while y < 1.0 - s/2 and len(centers) < n:
            x = s/2
            while x < 1.0 - s/2 and len(centers) < n:
                centers.append([x, y])
                x += s
            y += s
        while len(centers) < n:
            centers.append([0.5, 0.5])
        centers = np.array(centers[:n])
        centers += rng_init.normal(0, 0.006, centers.shape)
        centers = np.clip(centers, 0.06, 0.94)
        r = np.full(n, s * 0.38) + rng_init.normal(0, 0.004, n)
        r = np.clip(r, 0.03, 0.45)
        configs.append(np.column_stack([centers, r]).flatten())
        
    # 3. Random feasible placements with spacing heuristic
    for _ in range(12):
        centers = rng_init.uniform(0.15, 0.85, (n, 2))
        r = np.full(n, 0.06) + rng_init.uniform(-0.01, 0.01, n)
        r = np.clip(r, 0.02, 0.4)
        configs.append(np.column_stack([centers, r]).flatten())

    # Stage 1: Broad search
    for x0 in configs:
        run_opt(x0, iters=4000)
        
    # Stage 2: Basin hopping & local refinement on best
    if best_x is not None:
        # Fine perturbations
        for _ in range(25):
            rng = np.random.RandomState(_)
            x_p = best_x.copy()
            x_p[:2*n] += rng.normal(0, 0.0015, 2*n)
            x_p[:2*n] = np.clip(x_p[:2*n], 0.01, 0.99)
            run_opt(x_p, iters=3000)
            
        # Medium jumps to escape basins
        for _ in range(20):
            rng = np.random.RandomState(_ + 100)
            x_p = best_x.copy()
            noise_scale = rng.uniform(0.008, 0.025)
            x_p[:2*n] += rng.normal(0, noise_scale, 2*n)
            x_p[:2*n] = np.clip(x_p[:2*n], 0.05, 0.95)
            run_opt(x_p, iters=4000)
            
        # Radius scaling exploration
        for _ in range(15):
            rng = np.random.RandomState(_ + 200)
            x_p = best_x.copy()
            factor = 1.0 + rng.uniform(-0.12, 0.12)
            x_p[2*n:] *= factor
            x_p[2*n:] = np.clip(x_p[2*n:], 0.005, 0.5)
            run_opt(x_p, iters=3500)

    # Fallback safety
    if best_x is None:
        best_x = configs[0]
        
    cx = best_x[0::3]
    cy = best_x[1::3]
    r = best_x[2::3]
    
    centers = np.column_stack((cx, cy))
    radii = np.maximum(r, 0.0)
    
    # Strict post-processing to guarantee validity
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-9)
            
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < radii[i] + radii[j] - 1e-10:
                    excess = radii[i] + radii[j] - d
                    radii[i] -= excess / 2.0
                    radii[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
