# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000054 (state 91c332d2) state=b6b33190 sum of radii=2.625637 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

# Precompute indices for pairwise constraints to speed up evaluation
TRIU_IDX_I, TRIU_IDX_J = np.triu_indices(N, k=1)

def compute_constraints(v):
    """
    Computes boundary and non-overlap constraints.
    Returns a 1D array where all elements must be >= 0 for feasibility.
    """
    centers = v[:2 * N].reshape(N, 2)
    radii = v[2 * N:]
    
    con = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con.append(centers[:, 0] - radii)
    con.append(1.0 - centers[:, 0] - radii)
    con.append(centers[:, 1] - radii)
    con.append(1.0 - centers[:, 1] - radii)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    # Vectorized distance calculation
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangle constraints
    con.append(dists[TRIU_IDX_I, TRIU_IDX_J] - radii_sum[TRIU_IDX_I, TRIU_IDX_J])
    
    return np.concatenate(con)

def objective_func(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2 * N:])

def compute_initial_radii(centers):
    """Computes maximum feasible radii for a fixed set of centers."""
    x, y = centers[:, 0], centers[:, 1]
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Radius limited by half distance to nearest neighbor
    r_pair = 0.5 * np.min(dist, axis=1)
    
    return np.minimum(r_bound, r_pair)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-7, 0.5)] * N
    constraints = {'type': 'ineq', 'fun': compute_constraints}
    
    best_v = None
    best_sum = -np.inf
    
    # Helper to run optimization from a given start
    def run_optimizer(v0):
        try:
            res = minimize(
                objective_func, v0, method='SLSQP', bounds=bounds,
                constraints=constraints, 
                options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
            )
            curr_sum = -res.fun
            # Verify feasibility with a small tolerance
            c_vals = compute_constraints(res.x)
            if np.min(c_vals) > -1e-6:
                return res.x, curr_sum
        except Exception:
            pass
        return None, -np.inf

    # Generate diverse initial configurations
    starts = []
    
    # 1. Hexagonal lattice patterns
    for r0 in [0.08, 0.09, 0.095, 0.10]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N:
            x = r0 if row % 2 == 0 else 2 * r0
            while x + r0 <= 1.0 and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        starts.append(np.array(pts[:N]))
        
    # 2. Square grid patterns
    for r0 in [0.08, 0.09, 0.095]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N:
            x = r0
            while x + r0 <= 1.0 and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r0
            y += 2.0 * r0
            row += 1
        starts.append(np.array(pts[:N]))
        
    # 3. Dense random configurations
    for _ in range(8):
        starts.append(np.clip(np.random.rand(N, 2), 0.1, 0.9))
        
    # 4. Corner-focused configurations
    for _ in range(5):
        c = np.clip(np.random.rand(N, 2), 0.15, 0.85)
        c[:4] = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        starts.append(c)
        
    # Phase 1: Multi-start optimization
    for base_c in starts:
        # Add small perturbation to break symmetry
        c_pert = base_c + np.random.normal(0, 0.003, base_c.shape)
        c_pert = np.clip(c_pert, 1e-5, 1.0 - 1e-5)
        
        # Compute strictly feasible initial radii
        r_init = compute_initial_radii(c_pert) * 0.95
        r_init = np.maximum(r_init, 1e-6)
        
        v0 = np.concatenate([c_pert.flatten(), r_init])
        v_opt, s_val = run_optimizer(v0)
        
        if s_val > best_sum:
            best_sum = s_val
            best_v = v_opt
            
    # Phase 2: Iterative perturbation search to escape local minima
    if best_v is not None:
        curr_v = best_v.copy()
        for step in range(40):
            noise_scale = 0.004 * (0.9 ** step)
            v_pert = curr_v + np.random.normal(0, noise_scale, curr_v.shape)
            v_pert = np.clip(v_pert, 1e-5, 1.0 - 1e-5)
            v_pert[2 * N:] = np.clip(v_pert[2 * N:], 1e-6, 0.5)
            
            v_opt, s_val = run_optimizer(v_pert)
            if s_val > best_sum:
                best_sum = s_val
                best_v = v_opt
                curr_v = best_v
                
    # Extract final centers and radii
    centers = best_v[:2 * N].reshape(N, 2)
    radii = best_v[2 * N:].copy()
    
    # Phase 3: Strict validation repair
    for _ in range(30):
        valid = True
        
        # Fix boundary violations
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r_bound = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r_bound + 1e-12:
                radii[i] = max_r_bound
                valid = False
                
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    valid = False
                    
        radii = np.maximum(radii, 0.0)
        if valid:
            break
            
    return centers, radii, float(np.sum(radii))
