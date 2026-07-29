# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=01caf79c sum of radii=2.627847 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_pair_indices(n):
    """Precompute indices for all unique circle pairs to speed up constraint evaluation."""
    i_idx = []
    j_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx), np.array(j_idx)

def objective(v, n):
    """Objective: Minimize negative sum of radii (equivalent to maximizing sum of radii)."""
    return -np.sum(v[2*n:])

def constraints(v, n, pi, pj):
    """Compute inequality constraints: boundaries and non-overlap. All must be >= 0."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = np.concatenate([
        c[:, 0] - r,
        1 - c[:, 0] - r,
        c[:, 1] - r,
        1 - c[:, 1] - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized distance squared computation
    dist_sq = np.sum((c[pi] - c[pj])**2, axis=1)
    r_sum = r[pi] + r[pj]
    
    return np.concatenate([cons, dist_sq - r_sum**2])

def generate_initial_guesses(n):
    """Generate multiple diverse initial configurations."""
    guesses = []
    
    # 1. Hexagonal lattice
    pts = []
    y = 0.1
    row = 0
    while len(pts) < n + 10:
        x = 0.1 + (row % 2) * 0.085
        while x < 0.92 and len(pts) < n + 10:
            pts.append([x, y])
            x += 0.17
        y += 0.15
        row += 1
    guesses.append(np.array(pts[:n]))

    # 2. Rotated Hexagonal lattices (often better for boundary fitting)
    base_pts = pts[:n+10]
    for angle in [0.15, 0.3, 0.45]:
        pts_rot = []
        for p in base_pts:
            xr = p[0] * np.cos(angle) - p[1] * np.sin(angle) + 0.5
            yr = p[0] * np.sin(angle) + p[1] * np.cos(angle) + 0.5
            if 0.05 < xr < 0.95 and 0.05 < yr < 0.95:
                pts_rot.append([xr, yr])
        if len(pts_rot) >= n:
            guesses.append(np.array(pts_rot[:n]))

    # 3. Standard Grid
    pts_grid = []
    for i in range(6):
        for j in range(5):
            if len(pts_grid) < n:
                pts_grid.append([0.1 + i*0.16, 0.1 + j*0.2])
    guesses.append(np.array(pts_grid))
    
    # 4. Random dense placements
    np.random.seed(42)
    for _ in range(4):
        guesses.append(np.random.uniform(0.1, 0.9, size=(n, 2)))
        
    return guesses

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pi, pj = compute_pair_indices(n)
    
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)]*(2*n) + [(0.0, 0.5)]*n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n, pi, pj)}
    
    best_v = None
    best_val = -np.inf
    
    # Phase 1: Multi-start optimization from structured & random guesses
    initial_guesses = generate_initial_guesses(n)
    
    for pts in initial_guesses:
        # Add slight random jitter to break symmetry
        pts = pts + np.random.uniform(-0.005, 0.005, size=pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        r0 = np.full(n, 0.08)
        v0 = np.concatenate([pts.flatten(), r0])
        
        try:
            res = minimize(objective, v0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-11, 'disp': False})
            if -res.fun > best_val:
                best_val = -res.fun
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        # Fallback initialization if all optimizations fail
        best_v = np.zeros(3*n)
        best_v[:2*n] = np.random.uniform(0.2, 0.8, 2*n)
        best_v[2*n:] = 0.02
        
    # Phase 2: Local perturbation search to escape local minima
    current_v = best_v
    for step in range(6):
        # Slightly shrink radii to create breathing room
        scale = 0.996 - step * 0.001
        current_v[2*n:] *= scale
        
        # Perturb centers and radii
        noise = np.random.uniform(-0.004, 0.004, size=current_v.shape)
        noise[2*n:] *= 0.2  # Smaller noise for radii
        
        perturbed_v = np.clip(current_v + noise, 0.0, 1.0)
        perturbed_v[2*n:] = np.clip(perturbed_v[2*n:], 0.0, 0.5)
        
        try:
            res = minimize(objective, perturbed_v, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-11, 'disp': False})
            if -res.fun > best_val:
                best_val = -res.fun
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Extract final configuration
    centers = best_v[:2*n].reshape(n, 2)
    radii = best_v[2*n:]
    
    # Phase 3: Strict validity enforcement to satisfy validator tolerances
    # 1. Enforce boundary constraints strictly
    for i in range(n):
        max_r_bound = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        radii[i] = min(radii[i], max_r_bound)
        radii[i] = max(radii[i], 0.0)
        
    # 2. Enforce non-overlap strictly with safety margin
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            if d < radii[i] + radii[j] - 1e-9:
                shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-7
                radii[i] = max(radii[i] - shrink, 0.0)
                radii[j] = max(radii[j] - shrink, 0.0)
                
    return centers, radii, float(np.sum(radii))
