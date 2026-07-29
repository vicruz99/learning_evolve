# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=fae1ec65 sum of radii=2.617835 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def make_initial_config(seed):
    """Generate a diverse initial configuration based on seed."""
    rng = np.random.RandomState(seed)
    
    # Randomly choose between hexagonal lattice, perturbed grid, or repelled scatter
    layout_type = seed % 3
    
    if layout_type == 0:
        # Hexagonal lattice with random parameters
        r0 = 0.085 + rng.uniform(-0.01, 0.02)
        pts = []
        y = r0 + rng.uniform(-0.01, 0.01)
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + rng.uniform(-0.01, 0.01) + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
        pts = np.array(pts[:N])
    elif layout_type == 1:
        # Perturbed square grid
        grid_x = np.linspace(0.1, 0.9, 6)
        grid_y = np.linspace(0.1, 0.9, 5)
        cx, cy = np.meshgrid(grid_x, grid_y)
        pts = np.column_stack([cx.flatten(), cy.flatten()])[:N]
        pts += rng.uniform(-0.02, 0.02, pts.shape)
    else:
        # Random dense scatter with repulsion relaxation
        pts = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(100):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    dx = pts[i,0] - pts[j,0]
                    dy = pts[i,1] - pts[j,1]
                    d = np.hypot(dx, dy)
                    if d < 0.15 and d > 1e-4:
                        f = (0.15 - d) * 2.0 / d
                        forces[i] += f * np.array([dx, dy])
                        forces[j] -= f * np.array([dx, dy])
                for dim in range(2):
                    if pts[i,dim] < 0.05: forces[i,dim] += (0.05 - pts[i,dim]) * 10.0
                    if pts[i,dim] > 0.95: forces[i,dim] -= (pts[i,dim] - 0.95) * 10.0
            pts += forces * 0.02
            pts = np.clip(pts, 0.02, 0.98)

    # Apply random rotation to break symmetry for lattice/grid types
    if layout_type != 2:
        angle = rng.uniform(-0.25, 0.25)
        cx_c, cy_c = 0.5, 0.5
        pts[:, 0] -= cx_c
        pts[:, 1] -= cy_c
        c, s = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c, -s], [s, c]])
        pts[:, 0] += cx_c
        pts[:, 1] += cy_c

    pts = np.clip(pts, 0.02, 0.98)
    
    # Compute strictly feasible initial radii
    r_init = np.full(N, 0.5)
    for i in range(N):
        r_init[i] = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        for j in range(N):
            if i != j:
                d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                val = d / 2.0
                if val < r_init[i]: r_init[i] = val
                if val < r_init[j]: r_init[j] = val
    r_init *= 0.75 # Leave room for optimizer to expand
    
    return np.concatenate([pts[:,0], pts[:,1], r_init])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start optimization from diverse basins of attraction
    for seed in range(35):
        v0 = make_initial_config(seed)
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                # Verify feasibility with tolerance
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback initialization if optimization fails completely
    if best_v is None:
        best_v = make_initial_config(0)
        
    # Phase 2: Local refinement to escape shallow local minima
    current_v = best_v
    for step in range(20):
        # Gradually shrink radii to unstick circles from local equilibria
        scale = 0.975 - step * 0.001
        v_pert = current_v.copy()
        v_pert[2*N:] *= max(0.85, scale)
        
        # Perturb centers to explore neighborhood
        v_pert[:2*N] += np.random.uniform(-0.005, 0.005, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            pass
            
    # Extract optimal configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        cr[i] = min(cr[i], centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, cr, float(np.sum(cr))
