# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000034 (state 766fe0af) state=923014d5 sum of radii=2.617553 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for overlap constraints
TRI_U_IDX = np.triu_indices(N, k=1)

def compute_constraints(vars_flat):
    """
    Computes all boundary and non-overlap constraints.
    Returns an array where each element must be >= 0.
    Uses squared distances for better smoothness in optimization.
    """
    X = vars_flat.reshape(N, 3)
    xs, ys, rs = X[:, 0], X[:, 1], X[:, 2]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    idx_i, idx_j = TRI_U_IDX
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dr = rs[idx_i] + rs[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(c)

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def force_relax(centers, radii):
    """
    Vectorized force-directed relaxation to resolve overlaps and expand radii.
    """
    centers = centers.copy()
    radii = radii.copy()
    velocities = np.zeros_like(centers)
    
    # Simulation parameters
    dt = 0.006
    damping = 0.85
    k_rep = 180.0
    k_bound = 250.0
    max_iter = 2500
    
    for _ in range(max_iter):
        # Pairwise repulsion
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        dists = np.clip(dists, 1e-9, None)
        min_dists = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, min_dists - dists)
        dirs = diffs / dists[:, :, np.newaxis]
        forces = np.sum(overlap[:, :, np.newaxis] * dirs * k_rep, axis=1)
        
        # Boundary repulsion
        for dim in range(2):
            mask_low = centers[:, dim] < radii
            forces[mask_low, dim] += k_bound * (radii[mask_low] - centers[mask_low, dim])
            mask_high = centers[:, dim] > 1.0 - radii
            forces[mask_high, dim] -= k_bound * (centers[mask_high, dim] - (1.0 - radii[mask_high]))
            
        velocities = damping * velocities + forces * dt
        centers += velocities
        centers = np.clip(centers, 0.0, 1.0)
        
        # Expand radii when near equilibrium
        if np.max(np.abs(forces)) < 1e-4:
            radii *= 1.00025
            velocities *= 0.5
            
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    best_val = -np.inf
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # ---------------------------------------------------------
    # 1. Generate Diverse Initial Configurations
    # ---------------------------------------------------------
    inits = []
    
    # A. Hexagonal lattice variations
    for s in range(8):
        rng = np.random.default_rng(s)
        pts = []
        r0 = 0.092
        y = r0
        row = 0
        while len(pts) < N:
            x = r0 if row % 2 == 0 else 2 * r0
            s_off = rng.uniform(-0.005, 0.005, 2)
            while x <= 1.0 - r0 and len(pts) < N:
                pts.append([x + s_off[0], y + s_off[1]])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        inits.append((np.array(pts[:N]), np.full(N, r0)))
        
    # B. Perturbed Grid
    gx = np.linspace(0.11, 0.89, 6)
    gy = np.linspace(0.11, 0.89, 5)
    cx, cy = np.meshgrid(gx, gy)
    grid_pts = np.column_stack((cx.ravel(), cy.ravel()))[:N]
    inits.append((grid_pts + np.random.normal(0, 0.008, grid_pts.shape), np.full(N, 0.082)))
    
    # C. Dense Random
    for s in range(5):
        rng = np.random.default_rng(s + 50)
        pts = rng.uniform(0.12, 0.88, (N, 2))
        inits.append((pts, np.full(N, 0.045)))
        
    # ---------------------------------------------------------
    # 2. Force Relaxation & SLSQP Refinement
    # ---------------------------------------------------------
    for c_init, r_init in inits:
        c_rel, r_rel = force_relax(c_init, r_init)
        
        x0 = np.zeros(N * 3)
        x0[0::3] = c_rel[:, 0]
        x0[1::3] = c_rel[:, 1]
        x0[2::3] = r_rel
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
            if np.all(compute_constraints(res.x) >= -1e-7):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        best_x = inits[0][0].flatten()
        best_x[2::3] = 0.05
        
    # ---------------------------------------------------------
    # 3. Iterative "Grow and Push" to escape local minima
    # ---------------------------------------------------------
    curr_x = best_x.copy()
    for step in range(14):
        # Grow radii slightly
        rs = curr_x[2::3].copy()
        rs *= 1.0012 
        curr_x[2::3] = rs
        
        # Perturb centers to help resolve new overlaps
        curr_x[:2*N] += np.random.normal(0, 0.0004, 2*N)
        curr_x[:2*N] = np.clip(curr_x[:2*N], 0.0, 1.0)
        
        try:
            res = minimize(compute_objective, curr_x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-13})
            if np.all(compute_constraints(res.x) >= -1e-7):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
                    curr_x = best_x.copy()
        except Exception:
            pass
            
    # ---------------------------------------------------------
    # 4. Final Safety Repair
    # ---------------------------------------------------------
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x.reshape(N, 3)[:, 2].copy()
    
    # Deterministic shrink to strictly satisfy validator tolerance
    for _ in range(15):
        changed = False
        for i in range(N):
            # Boundary clamping
            mx = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
            if radii[i] > mx + 1e-9:
                radii[i] = mx
                changed = True
                
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
