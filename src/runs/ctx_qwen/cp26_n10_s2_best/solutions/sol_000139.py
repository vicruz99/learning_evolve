# sol_000139 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=ccb06e61 sum of radii=2.624472 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared for smooth gradients)."""
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

def relax_centers(centers, iterations=500, dt=0.0005):
    """Force-directed relaxation to spread centers apart (simulates equal circle packing)."""
    pts = centers.copy()
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 0.20 and d > 1e-6:
                    repulsion = (0.20 - d) / d * diff
                    forces[i] += repulsion
                    forces[j] -= repulsion
        # Boundary repulsion
        mask_l = pts[:, 0] < 0.05; forces[mask_l, 0] += 0.01
        mask_r = pts[:, 0] > 0.95; forces[mask_r, 0] -= 0.01
        mask_b = pts[:, 1] < 0.05; forces[mask_b, 1] += 0.01
        mask_t = pts[:, 1] > 0.95; forces[mask_t, 1] -= 0.01
        
        pts += forces * dt
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def get_feasible_radii(centers):
    """Compute safe initial radii based on local geometry."""
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    wall_dists = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    return np.clip(0.85 * np.minimum(min_dists / 2.0, wall_dists), 0.005, 0.25)

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Generate diverse initial configurations
    inits = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with rotations and shifts
    for seed in range(20):
        np.random.seed(seed)
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        shift = np.random.uniform(-0.05, 0.05, 2)
        pts = []
        r0 = 0.10 + np.random.uniform(-0.02, 0.02)
        for i in range(-5, 10):
            for j in range(-5, 10):
                x = i * r0 + (j % 2) * r0 * 0.5
                y = j * r0 * np.sqrt(3) * 0.5
                pts.append([x, y])
        pts = np.array(pts)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
        pts -= pts.mean(axis=0)
        pts += [0.5 + shift[0], 0.5 + shift[1]]
        mask = (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)
        if np.sum(mask) >= N:
            inits.append(pts[mask][:N])

    # 2. Grid variations
    for seed in range(10):
        np.random.seed(seed + 100)
        pts = np.array([[0.1 + i*0.16 + np.random.uniform(-0.02, 0.02), 
                         0.1 + j*0.20 + np.random.uniform(-0.02, 0.02)] 
                        for i in range(6) for j in range(5)][:N])
        inits.append(pts)

    # 3. Random dense scatter
    for seed in range(10):
        np.random.seed(seed + 200)
        inits.append(np.random.uniform(0.15, 0.85, size=(N, 2)))

    # Phase 1: Relax centers and optimize radii
    for centers in inits:
        centers_relaxed = relax_centers(centers)
        r_init = get_feasible_radii(centers_relaxed)
        v0 = np.concatenate([centers_relaxed[:, 0], centers_relaxed[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue

    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        current_v = best_v
        for step in range(15):
            np.random.seed(step + 1000)
            v_pert = current_v.copy()
            
            # Shrink radii to allow center movement
            scale = 0.92 - step * 0.01
            v_pert[2*N:] *= max(0.7, scale)
            
            # Perturb centers
            noise = np.random.uniform(-0.004, 0.004, 2*N)
            v_pert[:2*N] += noise
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                if -res.fun > best_sum:
                    if np.min(constraints(res.x)) >= -1e-6:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        current_v = best_v
            except Exception:
                continue

    # Extract and post-process
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict boundary enforcement
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # Strict non-overlap enforcement
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
