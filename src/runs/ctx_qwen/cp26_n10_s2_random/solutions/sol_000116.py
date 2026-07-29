# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000076 (state b16097a6) state=0db91b80 sum of radii=2.401875 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_radii(centers):
    """Computes maximum valid radii for a fixed set of centers."""
    n = centers.shape[0]
    # Distance to boundaries
    xb = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    yb = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    r_bound = np.minimum(xb, yb)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair)

def objective(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    centers = v.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)

def get_hex_init(n, spacing, shift_x=0.0, shift_y=0.0):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = spacing * 0.5 + shift_y
    row = 0
    while len(pts) < n:
        x_start = spacing * 0.5 + (spacing if row % 2 == 1 else 0.0) + shift_x
        x = x_start
        while x < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    return np.array(pts[:n])

def repulsion_init(n, iters=300, rng=None):
    """Generates a configuration using force-directed repulsion."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.uniform(0.1, 0.9, (n, 2))
    for _ in range(iters):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        forces = diff / (dists**2 + 1e-5)[:, :, np.newaxis]
        pts += 0.008 * np.sum(forces, axis=1)
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n)
    best_val = np.inf
    best_centers = None
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Hexagonal lattices with various densities and shifts
    for spacing in np.linspace(0.13, 0.26, 8):
        for sx in [0.0, 0.03, 0.06]:
            for sy in [0.0, 0.03, 0.06]:
                try:
                    c = get_hex_init(n, spacing, sx, sy)
                    c += rng.normal(0, 0.002, c.shape)
                    candidates.append(np.clip(c, 0.0, 1.0).flatten())
                except Exception:
                    pass

    # 2. Force-directed repulsion inits
    for _ in range(12):
        candidates.append(repulsion_init(n, rng=rng).flatten())

    # 3. Grid inits
    for side in range(4, 7):
        grid = np.linspace(0.08, 0.92, side)
        cx, cy = np.meshgrid(grid, grid)
        pts = np.column_stack((cx.flatten(), cy.flatten()))[:n]
        pts += rng.normal(0, 0.015, pts.shape)
        candidates.append(np.clip(pts, 0.0, 1.0).flatten())

    # 4. Corner-focused inits
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (n, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        candidates.append(np.clip(c, 0.0, 1.0).flatten())

    # Primary Optimization Phase
    for x0 in candidates:
        try:
            res = minimize(objective, x0, method='Powell', bounds=bounds,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'xtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(-1, 2)
        except Exception:
            continue

    # Iterative Refinement to escape local minima
    if best_centers is not None:
        curr = best_centers.copy()
        for i in range(40):
            sigma = 0.006 * (0.85 ** i)
            pert = curr + rng.normal(0, sigma, curr.shape)
            pert = np.clip(pert, 0.0, 1.0)
            try:
                res = minimize(objective, pert.flatten(), method='Powell', bounds=bounds,
                               options={'maxiter': 5000, 'ftol': 1e-13, 'xtol': 1e-13})
                if res.fun < best_val:
                    best_val = res.fun
                    best_centers = res.x.reshape(-1, 2)
            except Exception:
                continue
                
        # Final polish attempt
        try:
            res = minimize(objective, best_centers.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(-1, 2)
        except Exception:
            pass

    final_centers = best_centers if best_centers is not None else np.random.rand(n, 2)
    final_radii = compute_radii(final_centers)
    final_radii = np.maximum(final_radii, 0.0)
    final_sum = float(np.sum(final_radii))
    
    return final_centers, final_radii, final_sum
