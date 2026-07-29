# sol_000151 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000095 (state c7e336c8) state=10a35c03 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for vectorized constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Compute inequality constraints: boundaries and non-overlap.
    Uses squared distances for smoother gradients and better numerical stability.
    All elements must be >= 0.
    """
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
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    r_sum = r[PAIR_I] + r[PAIR_J]
    c = np.concatenate([c, dx**2 + dy**2 - r_sum**2])
    
    return c

def generate_repelled_centers(seed, n_iters=400):
    """Generate a well-spaced initial configuration using repulsive forces."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.2, 0.8, (N, 2))
    
    for _ in range(n_iters):
        forces = np.zeros_like(pts)
        # Vectorized pairwise repulsion
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        repulsion = np.zeros_like(dists)
        mask = dists < 0.3
        np.putmask(repulsion, mask, 0.5 / dists[mask])
        
        forces = np.sum(diff * repulsion[:, :, None], axis=1)
        
        # Boundary repulsion
        for dim in range(2):
            low_mask = pts[:, dim] < 0.15
            high_mask = pts[:, dim] > 0.85
            forces[low_mask, dim] += 0.8
            forces[high_mask, dim] -= 0.8
            
        pts += forces * 0.015
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def generate_hex_lattice(r0, shift_x, shift_y, rot_angle):
    """Generate hexagonal lattice centers with optional rotation."""
    pts = []
    y = r0 + shift_y
    row = 0
    while len(pts) < N + 8:
        x_start = r0 + shift_x + (row % 2) * r0
        x = x_start
        while x <= 1.0 - r0 and len(pts) < N + 8:
            pts.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
        row += 1
        
    pts = np.array(pts[:N])
    if abs(rot_angle) > 1e-6:
        cx, cy = 0.5, 0.5
        pts[:, 0] -= cx
        pts[:, 1] -= cy
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        pts = pts @ np.array([[c, -s], [s, c]])
        pts[:, 0] += cx
        pts[:, 1] += cy
        
    return np.clip(pts, 0.02, 0.98)

def compute_safe_radii(centers, scale=0.85):
    """Compute strictly feasible initial radii for given centers."""
    r = np.full(N, 0.4)
    # Boundary distances
    r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            half_d = d * 0.5
            if half_d < r[i]: r[i] = half_d
            if half_d < r[j]: r[j] = half_d
    return r * scale

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.45)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Diverse initial configurations
    inits = []
    
    # 1. Repulsion-based spreads
    for seed in range(25):
        inits.append(generate_repelled_centers(seed))
        
    # 2. Hexagonal lattices with variations
    for r0 in [0.085, 0.095, 0.105]:
        for sx in [-0.02, 0.0, 0.02]:
            for sy in [-0.02, 0.0, 0.02]:
                for ang in [0.0, 0.1, -0.1]:
                    inits.append(generate_hex_lattice(r0, sx, sy, ang))

    for c0 in inits:
        r0 = compute_safe_radii(c0, scale=0.82)
        v0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            # Verify feasibility with tolerance
            cons_vals = constraints(res.x)
            if np.min(cons_vals) >= -1e-7 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            continue

    # Phase 2: Basin-hopping refinement
    if best_v is not None:
        current_v = best_v
        for step in range(18):
            # Gradually shrink radii to unstick from local minima
            shrink = 0.97 - step * 0.008
            vp = current_v.copy()
            vp[2*N:] *= max(0.85, shrink)
            
            # Perturb centers to explore new equilibrium
            noise = np.random.uniform(-0.006, 0.006, 2*N)
            vp[:2*N] += noise
            vp[:2*N] = np.clip(vp[:2*N], 0.02, 0.98)
            
            try:
                res = minimize(objective, vp, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                
                curr_sum = -res.fun
                cons_vals = constraints(res.x)
                if np.min(cons_vals) >= -1e-7 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v
            except Exception:
                continue

    # Fallback (should not trigger)
    if best_v is None:
        c0 = generate_repelled_centers(0)
        r0 = compute_safe_radii(c0)
        best_v = np.concatenate([c0[:, 0], c0[:, 1], r0])
        
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing for validator compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap iteratively with minimal necessary shrinkage
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) * 0.5 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
