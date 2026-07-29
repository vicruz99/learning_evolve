# sol_000085 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000042 (state 26164787) state=5f173396 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[2*N:])

def constraints(vars):
    """
    Constraint function: ensures circles are inside the unit square and do not overlap.
    Returns a 1D array of constraint values that must be >= 0.
    """
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    c = []
    # Boundary constraints: circle inside [0,1]^2
    c.append(centers[:, 0] - radii)
    c.append(1.0 - centers[:, 0] - radii)
    c.append(centers[:, 1] - radii)
    c.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints: dist_sq >= (r_i + r_j)^2
    dx = centers[:, 0, np.newaxis] - centers[:, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    c.append(dist_sq[i_idx, j_idx] - r_sum[i_idx, j_idx]**2)
    
    return np.concatenate(c)

def generate_hex_init(row_counts, r_base, seed):
    """Generates a hexagonal lattice initialization with given row counts."""
    np.random.seed(seed)
    pts = []
    y = r_base
    for k, cnt in enumerate(row_counts):
        x_start = r_base if k % 2 == 0 else 2 * r_base
        for m in range(cnt):
            pts.append([x_start + m * 2 * r_base, y])
        y += np.sqrt(3) * r_base
        
    pts = np.array(pts[:N])
    
    # Add controlled perturbation
    pts += np.random.uniform(-0.025, 0.025, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    # Initialize radii with slight variation to break symmetry
    r_init = r_base + np.random.uniform(-0.005, 0.005, N)
    r_init = np.clip(r_init, 0.05, 0.2)
    
    return np.concatenate([pts.flatten(), r_init])

def generate_grid_init(seed):
    """Generates a 5x5 grid + 1 center initialization."""
    np.random.seed(seed)
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    pts.append([0.5, 0.5])
    pts = np.array(pts)
    
    pts += np.random.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    r_init = np.full(N, 0.09)
    r_init += np.random.uniform(-0.005, 0.005, N)
    r_init = np.clip(r_init, 0.05, 0.2)
    
    return np.concatenate([pts.flatten(), r_init])

def run_packing():
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -1.0
    
    # Collect diverse initializations
    inits = []
    
    # Hexagonal patterns with different row structures
    hex_patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [6, 6, 5, 5, 4],
        [4, 5, 6, 5, 6],
        [5, 5, 5, 5, 6]
    ]
    
    for pattern in hex_patterns:
        for seed in range(5):
            inits.append(generate_hex_init(pattern, 0.09, seed))
            
    # Grid patterns
    for seed in range(10):
        inits.append(generate_grid_init(seed))
        
    # Random but feasible starts
    np.random.seed(42)
    for seed in range(10):
        centers = np.random.uniform(0.1, 0.9, (N, 2))
        r = np.full(N, 0.08)
        inits.append(np.concatenate([centers.flatten(), r]))
        
    # Phase 1: Broad search
    for idx, x0 in enumerate(inits):
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            
            if res.success:
                cons_val = constraints(res.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = np.sum(res.x[2*N:])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: High-precision refinement on best and top alternatives
    if best_vars is not None:
        # Refine best
        try:
            res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 10000, 'ftol': 1e-15})
            if res_final.success:
                cons_val = constraints(res_final.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = np.sum(res_final.x[2*N:])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res_final.x.copy()
        except Exception:
            pass
            
        # Try small perturbations around best to escape local flat regions
        for p in range(8):
            np.random.seed(100 + p)
            x_pert = best_vars.copy()
            x_pert += np.random.normal(0, 0.002, x_pert.shape)
            x_pert[:2*N] = np.clip(x_pert[:2*N], 0.001, 0.999)
            x_pert[2*N:] = np.maximum(x_pert[2*N:], 1e-6)
            
            try:
                res_p = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 3000, 'ftol': 1e-14})
                if res_p.success:
                    cons_val = constraints(res_p.x)
                    if np.min(cons_val) >= -1e-9:
                        curr_sum = np.sum(res_p.x[2*N:])
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res_p.x.copy()
            except Exception:
                continue

    # Fallback
    if best_vars is None:
        best_vars = inits[0]
        
    # Post-processing: Ensure strict feasibility and maximize radii if possible
    centers = best_vars[:2*N].reshape(N, 2)
    radii = best_vars[2*N:]
    
    # Check and fix tiny violations
    cons_val = constraints(best_vars)
    if np.min(cons_val) < 0:
        # Slightly shrink radii to guarantee validity
        min_viol = np.min(cons_val)
        if min_viol < -1e-10:
            scale = 1.0 + min_viol * 0.01
            radii *= scale
            best_vars[2*N:] = radii
            centers = best_vars[:2*N].reshape(N, 2)
            
    # Final sum
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
