# sol_000110 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000076 (state b16097a6) state=9571ca7e sum of radii=1.880786 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_radii(centers):
    """
    Computes the maximum feasible radius for each circle given fixed centers.
    Ensures circles do not overlap and stay within the unit square.
    r_i = min(dist_to_boundary, 0.5 * dist_to_nearest_neighbor)
    """
    n = centers.shape[0]
    # Distance to boundaries
    xb = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    yb = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    r_bound = np.minimum(xb, yb)
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Radius constrained by half the minimum distance to avoid overlap
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair)

def objective(centers_flat):
    """Objective function: maximize sum of radii => minimize negative sum."""
    centers = centers_flat.reshape(-1, 2)
    # Clip to stay strictly inside the square to avoid boundary singularities
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    radii = compute_radii(centers)
    return -np.sum(radii)

def force_repulsion_init(rng, n, iterations=800):
    """Initializes centers using a simple force-directed repulsion to spread them out."""
    c = rng.uniform(0.2, 0.8, (n, 2))
    # Force corners to be utilized (high value for sum of radii)
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    c[:4] = corners
    
    for _ in range(iterations):
        forces = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = c[i] - c[j]
                dist = np.linalg.norm(d_vec)
                if dist < 0.2 and dist > 1e-5:
                    push = (0.2 - dist) * 0.1
                    f = (d_vec / dist) * push
                    forces[i] += f
                    forces[j] -= f
        c += forces
        c = np.clip(c, 0.01, 0.99)
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    bounds = [(0.0, 1.0)] * (2 * n)
    
    best_val = np.inf
    best_centers = None
    
    # --- Phase 1: Generate Diverse Initial Configurations ---
    candidates = []
    
    # 1. Force-repulsion initialized (handles corners and spread well)
    for _ in range(8):
        candidates.append(force_repulsion_init(rng, n))
        
    # 2. Hexagonal lattice patterns
    for r0 in [0.08, 0.09, 0.10]:
        c = []
        y = r0
        row = 0
        while len(c) < n:
            x = r0 if row % 2 == 0 else 2 * r0
            while x + r0 <= 1.0 and len(c) < n:
                c.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
            row += 1
        candidates.append(np.array(c[:n]))
        
    # 3. Square grids
    for s in [5, 6]:
        xs = np.linspace(0.15, 0.85, s)
        ys = np.linspace(0.15, 0.85, s)
        cx, cy = np.meshgrid(xs, ys)
        grid = np.column_stack((cx.flatten(), cy.flatten()))
        candidates.append(grid[:n])
        
    # 4. Dense random with jitter
    for _ in range(5):
        candidates.append(np.clip(rng.uniform(0.1, 0.9, (n, 2)) + rng.normal(0, 0.02, (n, 2)), 0.01, 0.99))

    # --- Phase 2: Multi-Start Optimization ---
    for c0 in candidates:
        c_jit = c0 + rng.normal(0, 0.002, c0.shape)
        c_jit = np.clip(c_jit, 0.001, 0.999)
        
        try:
            res = minimize(objective, c_jit.flatten(), method='Powell', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(n, 2)
        except Exception:
            continue

    # --- Phase 3: Perturbation & Swap Refinement ---
    # Escape local minima by perturbing and swapping positions
    for step in range(40):
        noise_scale = 0.003 * (0.9 ** (step // 5))
        
        # Perturbation
        c_pert = best_centers + rng.normal(0, noise_scale, best_centers.shape)
        c_pert = np.clip(c_pert, 0.001, 0.999)
        
        # Swap two random circles to break symmetries that stall optimizers
        idx = rng.choice(n, 2, replace=False)
        c_pert[idx] = c_pert[idx[::-1]]
        
        try:
            res = minimize(objective, c_pert.flatten(), method='Powell', bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(n, 2)
        except Exception:
            continue

    # --- Phase 4: Final Validation & Extraction ---
    radii = compute_radii(best_centers)
    
    # Safety repair for numerical precision guarantees
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < radii[i] + radii[j] - 1e-12:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return best_centers, radii, final_sum
