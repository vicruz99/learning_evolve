# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000036 (state ae916370) state=2cb48e83 sum of radii=2.611378 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_constraints(vars_flat):
    """Computes all boundary and non-overlap constraints for the packing."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]

    c = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dr = rs[idx_i] + rs[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(c)

def objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def get_bounds():
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def growing_init(seed):
    """Generates a dense, feasible initial configuration using force-directed growth."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.03)
    
    # Relaxation phase: grow radii and push apart
    for _ in range(800):
        radii *= 1.0005
        forces = np.zeros_like(centers)
        for i in range(N):
            for d in range(2):
                if centers[i,d] < radii[i]: 
                    forces[i,d] += 20.0 * (radii[i] - centers[i,d])
                if centers[i,d] > 1.0 - radii[i]: 
                    forces[i,d] -= 20.0 * (centers[i,d] - (1.0 - radii[i]))
            for j in range(i+1, N):
                dist = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if dist < radii[i] + radii[j] + 1e-6:
                    overlap = radii[i] + radii[j] + 1e-6 - dist
                    if dist > 1e-9:
                        f = 50.0 * overlap / dist
                        forces[i,0] += f * (centers[i,0] - centers[j,0])
                        forces[i,1] += f * (centers[i,1] - centers[j,1])
                        forces[j,0] -= f * (centers[i,0] - centers[j,0])
                        forces[j,1] -= f * (centers[i,1] - centers[j,1])
        centers += 0.005 * forces
        centers = np.clip(centers, 0.01, 0.99)
        
    # Compute strictly feasible radii for the relaxed centers
    for i in range(N):
        r_lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        for j in range(N):
            if i != j:
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                r_lim = min(r_lim, d - radii[j])
        radii[i] = max(1e-6, r_lim)
        
    vars = np.zeros(3*N)
    vars[0::3] = centers[:, 0]
    vars[1::3] = centers[:, 1]
    vars[2::3] = radii
    return vars

def refine_with_lp(centers):
    """Given fixed centers, maximize sum of radii exactly via Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i,j] = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i,j])
            
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        bounds_vals = [centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1]]
        for bv in bounds_vals:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bv)
            
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0.0, None))
    if res.success:
        return res.x
    return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_vars = None
    best_obj = -np.inf
    
    # Phase 1: Multi-start SLSQP from force-directed inits
    for seed in range(12):
        x0 = growing_init(seed)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            if np.all(compute_constraints(res.x) >= -1e-6):
                val = -res.fun
                if val > best_obj:
                    best_obj = val
                    best_vars = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Hybrid Local Search (Perturb -> LP Radii -> SLSQP Centers)
    if best_vars is not None:
        centers = best_vars.reshape(N, 3)[:, :2]
        radii = best_vars.reshape(N, 3)[:, 2]
        
        for iter_count in range(40):
            rng = np.random.default_rng(iter_count * 17 + 3)
            centers_pert = centers + rng.normal(0, 0.0015, (N, 2))
            centers_pert = np.clip(centers_pert, 0.02, 0.98)
            
            # Exact radius optimization for perturbed centers
            radii_lp = refine_with_lp(centers_pert)
            current_sum = np.sum(radii_lp)
            
            if current_sum > best_obj:
                best_obj = current_sum
                centers = centers_pert
                radii = radii_lp
                
                # Refine centers with updated radii
                x0 = np.zeros(3*N)
                x0[0::3] = centers[:, 0]
                x0[1::3] = centers[:, 1]
                x0[2::3] = radii
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
                    if np.all(compute_constraints(res.x) >= -1e-6):
                        val = -res.fun
                        if val > best_obj:
                            best_obj = val
                            best_vars = res.x.copy()
                            centers = best_vars.reshape(N, 3)[:, :2]
                            radii = best_vars.reshape(N, 3)[:, 2]
                except Exception:
                    pass

    # Fallback
    if best_vars is None:
        best_vars = growing_init(0)
        
    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    
    # Final strict validation adjustment
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-11
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mx + 1e-10:
                radii[i] = mx
                changed = True
        if not changed:
            break
            
    sum_r = float(np.sum(radii))
    return centers, radii, sum_r
