# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000076 (state b16097a6) state=9766af98 sum of radii=2.623075 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(v):
    """
    Computes boundary and non-overlap constraints.
    Returns a 1D array where all elements must be >= 0 for feasibility.
    """
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    
    con = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    idx = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
    con.append(d - (r[idx[0]] + r[idx[1]]))
    
    return np.concatenate(con)

def obj_func(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*N:])

def make_feasible(v):
    """
    Transforms an arbitrary parameter vector into a strictly feasible one.
    Clips centers to [0.02, 0.98] and computes safe radii.
    """
    c = np.clip(v[:2*N].reshape(N, 2), 0.02, 0.98)
    
    # Distance to boundaries
    rb = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    
    # Distance to nearest neighbor
    dists = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    
    # Take 95% of the maximum possible to ensure strict feasibility
    r = np.minimum(rb, rp) * 0.95
    r = np.maximum(r, 1e-6)
    
    return np.concatenate([c.flatten(), r])

def run_optimization(v0, rng):
    """Runs SLSQP optimization from a feasible starting point."""
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    try:
        res = minimize(obj_func, v0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 10000, 'ftol': 1e-15, 'disp': False})
        # Verify feasibility with tolerance
        c_vals = compute_constraints(res.x)
        if np.min(c_vals) >= -1e-8:
            return res.x, True
    except Exception:
        pass
    return v0, False

def generate_hex_init(r_est, rng):
    """Generates a hexagonal lattice initialization."""
    c = []
    y = r_est
    row = 0
    while len(c) < N:
        shift = r_est if row % 2 == 1 else 0.0
        x = r_est + shift
        while x + r_est <= 1.0 and len(c) < N:
            c.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3)
        row += 1
    return np.array(c[:N])

def generate_repulsion_init(rng):
    """Generates points spread by force-directed repulsion."""
    c = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(200):
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 0.15 and dist > 1e-8:
                    push = (0.15 - dist) * 0.5
                    c[i] += diff / dist * push
                    c[j] -= diff / dist * push
    return np.clip(c, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_v = None
    best_sum = -1.0
    
    # ---------------------------------------------------------
    # Phase 1: Multi-Start Optimization
    # ---------------------------------------------------------
    starts = []
    
    # Hexagonal patterns with varying densities
    for r_est in [0.09, 0.095, 0.10, 0.105]:
        hex_c = generate_hex_init(r_est, rng)
        hex_c += rng.normal(0, 0.005, hex_c.shape)
        starts.append(make_feasible(np.concatenate([hex_c.flatten(), np.zeros(N)])))
        
    # Force-directed random starts
    for _ in range(10):
        rep_c = generate_repulsion_init(rng)
        starts.append(make_feasible(np.concatenate([rep_c.flatten(), np.zeros(N)])))
        
    # Run optimization from each start
    for v0 in starts:
        v_opt, success = run_optimization(v0, rng)
        if success:
            s = np.sum(v_opt[2*N:])
            if s > best_sum:
                best_sum = s
                best_v = v_opt.copy()
                
    # Fallback if all fails
    if best_v is None:
        best_v = starts[0]
        best_sum = np.sum(best_v[2*N:])

    # ---------------------------------------------------------
    # Phase 2: Iterative Perturbation & Radius Expansion
    # ---------------------------------------------------------
    for step in range(35):
        # Decaying noise scale
        noise_scale = 0.006 * (0.88 ** step)
        v_pert = best_v + rng.normal(0, noise_scale, best_v.shape)
        
        # Enforce strict feasibility before optimization
        v_pert = make_feasible(v_pert)
        
        v_opt, success = run_optimization(v_pert, rng)
        if success:
            s = np.sum(v_opt[2*N:])
            if s > best_sum:
                best_sum = s
                best_v = v_opt.copy()
                
                # Try to scale up radii slightly and re-optimize
                # This directly pushes against the objective ceiling
                for _ in range(3):
                    v_push = best_v.copy()
                    v_push[2*N:] *= 1.0015  # Grow radii by 0.15%
                    v_push = make_feasible(v_push) # Fix centers to accommodate growth
                    v_opt2, success2 = run_optimization(v_push, rng)
                    if success2:
                        s2 = np.sum(v_opt2[2*N:])
                        if s2 > best_sum:
                            best_sum = s2
                            best_v = v_opt2.copy()
                        else:
                            break
                    else:
                        break

    # ---------------------------------------------------------
    # Phase 3: Strict Numerical Repair & Validation Guarantee
    # ---------------------------------------------------------
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:].copy()
    
    # Iteratively shrink overlapping or out-of-bound circles
    for _ in range(50):
        changed = False
        
        # Fix boundary violations
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r_bound = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r_bound + 1e-12:
                radii[i] = max_r_bound
                changed = True
                
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
