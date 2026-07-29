# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000056 (state 0fa800b4) state=9e9c9968 sum of radii=1.067581 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def compute_constraints(vars_flat):
    """
    Computes boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0 for feasibility.
    """
    x = vars_flat[0::3]
    y = vars_flat[1::3]
    r = vars_flat[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2  =>  dist^2 - (r_i + r_j)^2 >= 0
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_overlap = (dx**2 + dy**2 - dr**2)[mask]
    
    return np.concatenate([c_bound, c_overlap])

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return bounds

def solve_radii_lp(centers):
    """
    Given fixed centers, solves the LP to find radii that maximize sum(r_i)
    while satisfying all non-overlap and boundary constraints strictly.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    num_pairs = N * (N - 1) // 2
    num_constraints = num_pairs + 4 * N
    A = np.zeros((num_constraints, N))
    b = np.zeros(num_constraints)
    
    idx = 0
    # Pairwise constraints: r_i + r_j <= distance(i,j)
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            b[idx] = d - 1e-9  # Small safety margin
            idx += 1
            
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(N):
        for val in [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]:
            A[idx, i] = 1.0
            b[idx] = val - 1e-9
            idx += 1
            
    bounds_r = [(0.0, None)] * N
    res = linprog(-np.ones(N), A_ub=A, b_ub=b, bounds=bounds_r, method='highs')
    
    if res.success:
        return res.x
    return np.full(N, 1e-6)

def generate_hex_init(rows, seed):
    """Generates a hexagonal lattice initialization with specified row counts."""
    rng = np.random.default_rng(seed)
    pos = []
    r0 = 0.10
    y = r0
    for r_idx, count in enumerate(rows):
        x_start = r0 + (r_idx % 2) * r0
        for _ in range(count):
            pos.append([x_start, y])
            x_start += 2 * r0
        y += r0 * np.sqrt(3.0)
        
    pos = np.array(pos[:N]) + rng.normal(0, 0.004, (N, 2))
    return np.clip(pos, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_vars = None
    best_obj = -np.inf
    candidates = []
    
    # Diverse initial configurations
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6]
    ]
    for pat in row_patterns:
        for s in range(3):
            c = generate_hex_init(pat, s)
            v = np.zeros(N * 3)
            v[0::3] = c[:, 0]
            v[1::3] = c[:, 1]
            v[2::3] = 0.09
            candidates.append(v)
            
    # Random starts
    for s in range(12):
        c = np.random.rand(N, 2) * 0.8 + 0.1
        v = np.zeros(N * 3)
        v[0::3] = c[:, 0]
        v[1::3] = c[:, 1]
        v[2::3] = 0.07
        candidates.append(v)
        
    # Primary optimization phase
    for x0 in candidates:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13})
            
            c_vals = compute_constraints(res.x)
            if np.all(c_vals >= -1e-8):
                curr_obj = -res.fun
                if curr_obj > best_obj:
                    best_obj = curr_obj
                    best_vars = res.x.copy()
        except Exception:
            continue

    # Basin-hopping refinement to escape local minima
    if best_vars is not None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            x_curr = best_vars.copy()
            x_curr += rng.normal(0, 0.0008, x_curr.shape)
            x_curr = np.clip(x_curr, 0.0, 1.0)
            x_curr[2::3] = np.clip(x_curr[2::3], 1e-7, 0.5)
            
            try:
                res = minimize(compute_objective, x_curr, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
                
                c_vals = compute_constraints(res.x)
                if np.all(c_vals >= -1e-8):
                    curr_obj = -res.fun
                    if curr_obj > best_obj:
                        best_obj = curr_obj
                        best_vars = res.x.copy()
            except Exception:
                continue

    # Fallback if optimization failed entirely
    if best_vars is None:
        best_vars = candidates[0]

    # Extract centers and solve LP for maximally valid radii
    centers = best_vars[:2 * N].reshape(N, 2)
    radii = solve_radii_lp(centers)
    
    # Final strict safety clipping to guarantee validator tolerance
    radii = np.maximum(radii, 0.0)
    for i in range(N):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if radii[i] > max_r + 1e-10:
            radii[i] = max_r - 1e-10
            
    # Pairwise safety check
    for _ in range(5):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d + 1e-10) / 2.0
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
