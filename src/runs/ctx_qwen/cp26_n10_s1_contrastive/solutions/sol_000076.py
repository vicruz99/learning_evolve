# sol_000076 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=34ecb334 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def constraints(vars_vec):
    """Compute all inequality constraints: boundary containment and pairwise separation."""
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation constraints: dist_sq >= (r_i + r_j)^2
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    r_sum = r[I_IDX] + r[J_IDX]
    c.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c)

def get_feasible_radii(centers):
    """Compute strictly feasible initial radii for a given set of centers."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        d_wall = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        d_min = np.inf
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < d_min:
                    d_min = d
        # 0.4 factor ensures strict feasibility margin for optimizer start
        r[i] = 0.4 * min(d_wall, 0.5 * d_min)
    return r

def generate_init_hex(seed):
    """Generate a hexagonal lattice initialization with slight random perturbation."""
    np.random.seed(seed)
    rows_counts = [5, 6, 5, 6, 4]
    pts = []
    y = 0.06
    for r_idx, count in enumerate(rows_counts):
        shift = (r_idx % 2) * 0.1
        x = 0.06 + shift
        for _ in range(count):
            pts.append([x, y])
            x += 0.18
        y += 0.15
    pts = np.array(pts[:N])
    pts += np.random.uniform(-0.01, 0.01, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    radii = get_feasible_radii(pts)
    
    vars0 = np.zeros(3*N)
    vars0[0::3] = pts[:, 0]
    vars0[1::3] = pts[:, 1]
    vars0[2::3] = radii
    return vars0

def generate_init_force(seed, n_iters=200):
    """Generate a force-directed layout initialization."""
    rng = np.random.RandomState(seed)
    centers = rng.rand(N, 2)
    for _ in range(n_iters):
        forces = np.zeros((N, 2))
        # Wall repulsion
        for i in range(N):
            x, y = centers[i]
            if x < 0.05: forces[i, 0] += 0.05 - x
            elif x > 0.95: forces[i, 0] -= x - 0.95
            if y < 0.05: forces[i, 1] += 0.05 - y
            elif y > 0.95: forces[i, 1] -= y - 0.95
            
        # Pair repulsion
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff) + 1e-6
                f = 0.3 / (dist**2)
                forces[i] += f * diff
                forces[j] -= f * diff
                
        centers += forces * 0.01
        centers = np.clip(centers, 0.02, 0.98)
        
    radii = get_feasible_radii(centers)
    vars0 = np.zeros(3*N)
    vars0[0::3] = centers[:, 0]
    vars0[1::3] = centers[:, 1]
    vars0[2::3] = radii
    return vars0

def run_packing():
    best_sum = -np.inf
    best_x = None
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Phase 1: Broad search from diverse initializations
    inits = []
    for s in range(10):
        inits.append(generate_init_hex(s))
    for s in range(15):
        inits.append(generate_init_force(s))
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            if res.success:
                cons_val = constraints(res.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local perturbation refinement
    if best_x is not None:
        for _ in range(30):
            x_pert = best_x + np.random.randn(3*N) * 0.003
            x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
            x_pert[2::3] = np.clip(x_pert[2::3], 1e-6, 0.49)
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
                if res.success:
                    cons_val = constraints(res.x)
                    if np.min(cons_val) >= -1e-9:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_x = res.x.copy()
            except Exception:
                pass

    # Phase 3: High-precision final polish
    if best_x is not None:
        try:
            res_final = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14})
            if res_final.success:
                cons_val = constraints(res_final.x)
                if np.min(cons_val) >= -1e-10:
                    best_x = res_final.x
                    best_sum = -res_final.fun
        except Exception:
            pass

    # Fallback safety net
    if best_x is None:
        best_x = generate_init_hex(0)
        best_sum = -objective(best_x)

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
