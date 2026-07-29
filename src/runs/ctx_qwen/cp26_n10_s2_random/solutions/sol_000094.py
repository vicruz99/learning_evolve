# sol_000094 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000068 (state 22e68fa8) state=92cf8b97 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii to maximize total radius."""
    return -np.sum(v[2::3])

def constraints(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    X = x[:, None] - x[None, :]
    Y = y[:, None] - y[None, :]
    R = r[:, None] + r[None, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, (X**2 + Y**2)[mask] - R[mask]**2])
    return c

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def init_force_directed(seed, steps=300):
    """Generates initial configuration using repulsion-based layout."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, (N, 2))
    r = 0.08
    
    for _ in range(steps):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        safe_dist = np.where(dist < 1e-9, 1e-9, dist)
        force_vec = (diff / safe_dist[:, :, None]) * np.clip(2.0 * r - dist, 0.0, None)[:, :, np.newaxis]
        force = np.sum(force_vec, axis=1)
        
        centers += 0.008 * force
        centers = np.clip(centers, 0.05, 0.95)
        
    # Compute feasible initial radii
    rs = np.full(N, 0.05)
    for i in range(N):
        d_b = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        d_n = np.min(np.hypot(centers[i,0]-centers[mask,0], centers[i,1]-centers[mask,1]))
        rs[i] = min(d_b, d_n/2.0) * 0.95
        
    v = np.zeros(3 * N)
    v[0::3] = centers[:, 0]
    v[1::3] = centers[:, 1]
    v[2::3] = rs
    return v

def init_hex(seed, scale=1.0):
    """Generates initial configuration from a perturbed hexagonal lattice."""
    rng = np.random.default_rng(seed)
    centers = []
    r0 = 0.105 * scale
    y = r0
    row = 0
    while len(centers) < N:
        x_start = r0 if row % 2 == 0 else 2 * r0
        x = x_start
        while x + r0 <= 1.0 + 1e-9 and len(centers) < N:
            centers.append([x + rng.normal(0, 0.005), y + rng.normal(0, 0.005)])
            x += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
    while len(centers) < N:
        centers.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        
    centers = np.array(centers[:N])
    centers = np.clip(centers, 0.05, 0.95)
    
    rs = np.full(N, 0.05)
    for i in range(N):
        d_b = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        d_n = np.min(np.hypot(centers[i,0]-centers[mask,0], centers[i,1]-centers[mask,1]))
        rs[i] = min(d_b, d_n/2.0) * 0.95
        
    v = np.zeros(3 * N)
    v[0::3] = centers[:, 0]
    v[1::3] = centers[:, 1]
    v[2::3] = rs
    return v

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Main function to pack 26 circles in a unit square."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -np.inf
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    for s in range(15):
        inits.append(init_hex(s, scale=0.95))
        inits.append(init_hex(s, scale=1.0))
        inits.append(init_force_directed(s))
        
    # Phase 2: Multi-start optimization
    for i, v0 in enumerate(inits):
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            s_val = -res.fun
            if s_val > best_sum:
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6:
                    best_sum = s_val
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Local perturbation search to escape local minima
    if best_v is not None:
        for _ in range(50):
            v_trial = best_v.copy()
            v_trial += np.random.normal(0, 0.001, v_trial.shape)
            v_trial[0::3] = np.clip(v_trial[0::3], 0.02, 0.98)
            v_trial[1::3] = np.clip(v_trial[1::3], 0.02, 0.98)
            v_trial[2::3] = np.clip(v_trial[2::3], 0.01, 0.45)
            try:
                res = minimize(objective, v_trial, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
                if -res.fun > best_sum:
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-6:
                        best_sum = -res.fun
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # Fallback
    if best_v is None:
        best_v = init_hex(0)
        
    # Extract centers and radii
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3].copy()
    
    # Phase 4: Deterministic Repair for Strict Validation Compliance
    for _ in range(100):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Clamp to boundaries
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-9:
                radii[i] = mr
                changed = True
                
        if not changed:
            break
            
    # Ensure non-negativity
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
