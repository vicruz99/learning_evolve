# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b444b7b1) state=ce4af98f sum of radii=2.504982 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, linprog

N = 26
I_IDX, J_IDX = np.tril_indices(N, -1)

def obj(vars):
    return -np.sum(vars[2*N:])

def constr(vars):
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    # Boundary constraints: distance to edges - radius >= 0
    bnd = np.empty(4*N)
    bnd[0::4] = centers[:, 0] - radii
    bnd[1::4] = 1.0 - centers[:, 0] - radii
    bnd[2::4] = centers[:, 1] - radii
    bnd[3::4] = 1.0 - centers[:, 1] - radii
    
    # Pairwise constraints: distance between centers - sum of radii >= 0
    diffs = centers[I_IDX] - centers[J_IDX]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    pairs = dists - radii[I_IDX] - radii[J_IDX]
    
    return np.concatenate([bnd, pairs])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    nl_con = NonlinearConstraint(constr, 0, np.inf)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Multiple restarts from perturbed hexagonal lattices
    for trial in range(5):
        np.random.seed(trial * 100 + 42)
        pts = []
        y = 0.12
        row_idx = 0
        while len(pts) < N:
            x_start = 0.12 if row_idx % 2 == 0 else 0.20
            x = x_start
            while x < 0.90 and len(pts) < N:
                pts.append([x + np.random.normal(0, 0.005), y + np.random.normal(0, 0.005)])
                x += 0.18
            y += 0.16
            row_idx += 1
            
        centers0 = np.array(pts[:N])
        centers0 = np.clip(centers0, 0.05, 0.95)
        radii0 = np.full(N, 0.01)
        x0 = np.concatenate([centers0.flatten(), radii0])
        
        try:
            res = minimize(obj, x0, method='trust-constr', bounds=bounds, 
                           constraints=nl_con, options={'maxiter': 800, 'verbose': 0})
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = res.x[:2*N].reshape(N, 2)
                best_radii = res.x[2*N:]
        except Exception:
            continue
            
    if best_centers is None:
        centers0 = np.random.rand(N, 2) * 0.8 + 0.1
        best_centers = centers0
        best_radii = np.full(N, 0.01)
        best_sum = -np.sum(best_radii)
        
    # LP refinement: fix centers, optimally solve for radii
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    for i in range(N):
        x, y = best_centers[i]
        for b in [x, 1-x, y, 1-y]:
            row = np.zeros(N)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds_r = [(0, None)] * N
    
    try:
        lp_res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if lp_res.success:
            best_radii = lp_res.x
            best_sum = -np.sum(best_radii)
    except Exception:
        pass
        
    return best_centers, best_radii, float(best_sum)
