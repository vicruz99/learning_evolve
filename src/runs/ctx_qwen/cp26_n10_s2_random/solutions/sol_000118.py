# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000076 (state b16097a6) state=31242775 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)

def compute_safe_radii(centers):
    """Computes strictly feasible initial radii for a given set of centers."""
    xb = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    yb = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    r_bound = np.minimum(xb, yb)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair) * 0.95

def obj_func(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*N:])

def constraint_func(v):
    """Computes all boundary and non-overlap constraints. Returns array >= 0 for feasibility."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    
    con = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    diff = c[TRIU_I] - c[TRIU_J]
    d = np.sqrt(np.sum(diff**2, axis=1))
    con.append(d - (r[TRIU_I] + r[TRIU_J]))
    
    return np.concatenate(con)

def force_relax(centers, steps=400):
    """Spreads points apart using repulsive forces and boundary repulsion."""
    c = centers.copy()
    for t in range(steps):
        forces = np.zeros_like(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise repulsion
        mask = dists < 0.22
        safe_dists = np.where(mask, dists, 0.22)
        push = np.where(mask, (0.22 - dists) * (0.22 / safe_dists), 0.0)
        
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    f = push[i, j]
                    forces[i] += diff[i, j] * f
                    forces[j] -= diff[i, j] * f
                    
        # Boundary repulsion
        for i in range(N):
            for dim in range(2):
                if c[i, dim] < 0.06:
                    forces[i, dim] += (0.06 - c[i, dim]) * 15.0
                elif c[i, dim] > 0.94:
                    forces[i, dim] -= (c[i, dim] - 0.94) * 15.0
                    
        lr = 0.015 / (1.0 + t * 0.01)
        c += forces * lr
        c = np.clip(c, 0.02, 0.98)
    return c

def get_hex_init(pattern, r_est=0.095):
    """Generates a hexagonal lattice initialization based on row counts."""
    centers = []
    y = r_est
    for r_idx, cnt in enumerate(pattern):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            centers.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
    c = np.array(centers[:N])
    c += np.random.normal(0, 0.004, c.shape)
    return np.clip(c, 0.12, 0.88)

def get_corner_init(rng):
    """Generates an initialization biased towards corners and edges."""
    c = rng.uniform(0.15, 0.85, (N, 2))
    # Force 4 circles to corners
    corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
    c[:4] = corners
    # Force 8 circles to mid-edges
    edges = [[0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5],
             [0.25, 0.12], [0.75, 0.12], [0.25, 0.88], [0.75, 0.88]]
    c[4:12] = edges
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_v = None
    best_sum = -1.0
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal patterns
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [4, 6, 6, 6, 4], [5, 5, 6, 5, 5], [6, 4, 6, 5, 5],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6]
    ]
    for pat in patterns:
        c = get_hex_init(pat)
        c = force_relax(c, steps=200)
        r = compute_safe_radii(c)
        inits.append(np.concatenate([c.flatten(), r]))
        
    # 2. Random + Force Relaxation
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        c = force_relax(c, steps=300)
        r = compute_safe_radii(c)
        inits.append(np.concatenate([c.flatten(), r]))
        
    # 3. Corner/Edge biased
    for _ in range(5):
        c = get_corner_init(rng)
        c = force_relax(c, steps=250)
        r = compute_safe_radii(c)
        inits.append(np.concatenate([c.flatten(), r]))
        
    # Phase 1: Multi-start constrained optimization
    for v0 in inits:
        try:
            res = minimize(obj_func, v0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            c_vals = constraint_func(res.x)
            if np.min(c_vals) >= -1e-7:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue

    if best_v is None:
        best_v = inits[0]
        
    # Phase 2: Perturbation search to escape local minima
    for step in range(35):
        noise_scale = 0.005 * (0.88 ** step)
        v_pert = best_v + rng.normal(0, noise_scale, best_v.shape)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 1e-6, 0.45)
        
        try:
            res = minimize(obj_func, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
            c_vals = constraint_func(res.x)
            if np.min(c_vals) >= -1e-7:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Extract results
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict numerical repair to guarantee validation passes
    for _ in range(60):
        changed = False
        # Fix overlaps proportionally
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    overlap = req - d
                    shrink = overlap * 0.5 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
