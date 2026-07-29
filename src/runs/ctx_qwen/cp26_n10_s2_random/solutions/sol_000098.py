# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000083 (state 006ca278) state=1e3ffaf1 sum of radii=2.485479 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
_OVERLAP_I, _OVERLAP_J = np.triu_indices(N, k=1)

def compute_max_radii(centers):
    """Computes the maximum possible radius for each circle given fixed centers."""
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Radius limited by half the minimum distance to neighbors
    r_pair = 0.5 * np.min(dist, axis=1)
    
    return np.minimum(r_bound, r_pair)

def objective_centers(v):
    """Objective for centers-only optimization: maximize sum of radii."""
    return -np.sum(compute_max_radii(v.reshape(N, 2)))

def get_bounds_centers():
    """Bounds for center coordinates to keep them safely inside the square."""
    return [(1e-5, 1.0 - 1e-5)] * (2 * N)

def generate_hex_grid(r_init, shift_x=0.0, shift_y=0.0):
    """Generates a hexagonal lattice configuration perturbed by shifts."""
    centers = []
    y = r_init + shift_y
    row = 0
    while len(centers) < N:
        offset = (r_init if row % 2 == 1 else 0.0) + shift_x
        x = r_init + offset
        while x + r_init <= 1.0 + 1e-9 and len(centers) < N:
            centers.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3.0) * r_init
        row += 1
    return np.array(centers[:N])

def compute_constraints_full(vars):
    """Computes inequality constraints for joint SLSQP optimization."""
    c = vars[:2*N].reshape(N, 2)
    r = vars[2*N:]
    
    # Boundary constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    cons = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Overlap constraints: dist(i, j) >= r_i + r_j
    dx = c[:, 0, None] - c[:, 0]
    dy = c[:, 1, None] - c[:, 1]
    dists = np.sqrt(dx**2 + dy**2)
    cons = np.concatenate([cons, dists[_OVERLAP_I, _OVERLAP_J] - (r[_OVERLAP_I] + r[_OVERLAP_J])])
    return cons

def obj_full(vars):
    """Objective for joint optimization: maximize sum of radii."""
    return -np.sum(vars[2*N:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hybrid approach: Powell optimization on centers, followed by SLSQP joint polish.
    """
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_val = np.inf
    best_centers = None
    bounds_c = get_bounds_centers()
    starts = []
    
    # ---------------------------------------------------------
    # Phase 1: Generate Diverse Initial Configurations
    # ---------------------------------------------------------
    # Hexagonal lattices with various base radii and shifts
    for r_init in [0.09, 0.095, 0.10, 0.105, 0.11]:
        for sx in [0.0, 0.02, -0.02]:
            for sy in [0.0, 0.02, -0.02]:
                c = generate_hex_grid(r_init, sx, sy)
                c += rng.normal(0, 0.005, c.shape)
                c = np.clip(c, 1e-4, 1.0 - 1e-4)
                starts.append(c.flatten())
                
    # Perturbed 5x5 grid + 1 center
    for _ in range(10):
        gx = np.linspace(0.15, 0.85, 5)
        gy = np.linspace(0.15, 0.85, 5)
        cx, cy = np.meshgrid(gx, gy)
        c = np.column_stack((cx.flatten(), cy.flatten()))
        c = np.vstack([c, [0.5, 0.5]])
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 1e-4, 1.0 - 1e-4)
        starts.append(c.flatten())
        
    # Dense random starts
    for _ in range(20):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c.flatten())
        
    # ---------------------------------------------------------
    # Phase 2: Multi-Start Powell Optimization on Centers
    # ---------------------------------------------------------
    for x0 in starts:
        try:
            res = minimize(objective_centers, x0, method='Powell', bounds=bounds_c,
                           options={'maxiter': 3000, 'ftol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(N, 2)
        except Exception:
            continue
            
    # ---------------------------------------------------------
    # Phase 3: Basin Hopping Refinement
    # ---------------------------------------------------------
    if best_centers is not None:
        curr_c = best_centers.copy()
        for step in range(40):
            pert = curr_c + rng.normal(0, 0.005 * (0.9**step), curr_c.shape)
            pert = np.clip(pert, 1e-4, 1.0 - 1e-4)
            try:
                res = minimize(objective_centers, pert.flatten(), method='Powell', bounds=bounds_c,
                               options={'maxiter': 1500, 'ftol': 1e-14})
                if res.fun < best_val:
                    best_val = res.fun
                    best_centers = res.x.reshape(N, 2)
                    curr_c = best_centers.copy()
            except Exception:
                pass
                
    # ---------------------------------------------------------
    # Phase 4: SLSQP Joint Polish (Centers + Radii)
    # ---------------------------------------------------------
    r_init = compute_max_radii(best_centers)
    x0_full = np.concatenate([best_centers.flatten(), r_init])
    bounds_full = [(0.0, 1.0)] * (2*N) + [(1e-8, 0.5)] * N
    cons_full = {'type': 'ineq', 'fun': compute_constraints_full}
    
    try:
        res_full = minimize(obj_full, x0_full, method='SLSQP', bounds=bounds_full,
                            constraints=cons_full, options={'maxiter': 8000, 'ftol': 1e-15})
        if np.all(compute_constraints_full(res_full.x) >= -1e-9):
            best_centers = res_full.x[:2*N].reshape(N, 2)
            r_final = res_full.x[2*N:]
        else:
            r_final = compute_max_radii(best_centers)
    except Exception:
        r_final = compute_max_radii(best_centers)
        
    # ---------------------------------------------------------
    # Phase 5: Deterministic Repair for Strict Validation
    # ---------------------------------------------------------
    for _ in range(30):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_centers[i,0] - best_centers[j,0], best_centers[i,1] - best_centers[j,1])
                if d < r_final[i] + r_final[j] - 1e-12:
                    ov = r_final[i] + r_final[j] - d
                    r_final[i] -= ov / 2.0
                    r_final[j] -= ov / 2.0
                    changed = True
                    
        # Clamp to boundaries
        for i in range(N):
            mx = min(best_centers[i,0], 1.0 - best_centers[i,0], best_centers[i,1], 1.0 - best_centers[i,1])
            if r_final[i] > mx + 1e-12:
                r_final[i] = mx
                changed = True
                
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return best_centers, r_final, float(np.sum(r_final))
