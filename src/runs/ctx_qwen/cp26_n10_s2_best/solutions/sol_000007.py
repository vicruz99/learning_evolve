# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 77dfa116) state=bf6d1306 sum of radii=2.549783 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective function: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*N:3*N])

def constraints(v):
    """Compute all inequality constraints: must return values >= 0."""
    xs = v[:N]
    ys = v[N:2*N]
    rs = v[2*N:3*N]

    # Total constraints: 4 per circle (boundary) + N*(N-1)/2 (pairwise distance)
    num_cons = 4 * N + N * (N - 1) // 2
    cons = np.empty(num_cons)
    idx = 0

    # Boundary constraints
    for i in range(N):
        cons[idx] = xs[i] - rs[i]          # x >= r
        idx += 1
        cons[idx] = 1.0 - xs[i] - rs[i]    # x + r <= 1
        idx += 1
        cons[idx] = ys[i] - rs[i]          # y >= r
        idx += 1
        cons[idx] = 1.0 - ys[i] - rs[i]    # y + r <= 1
        idx += 1

    # Pairwise non-overlap constraints
    for i in range(N):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, N):
            dx = xi - xs[j]
            dy = yi - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            cons[idx] = d - rs[i] - rs[j]  # distance >= r_i + r_j
            idx += 1

    return cons

def run_packing():
    # 1. Initial configuration: grid layout with safe initial radii
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.full(N, 0.05)

    cols = 6
    rows = 5
    for i in range(N):
        c = i % cols
        r_idx = i // cols
        xs[i] = (c + 0.5) / cols
        ys[i] = (r_idx + 0.5) / rows

    # Flatten to optimization variable vector
    v0 = np.concatenate([xs, ys, rs])
    
    # Variable bounds
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N

    # Constraint dictionary
    cons_dict = {'type': 'ineq', 'fun': constraints}

    # 2. Optimize
    res = minimize(
        objective, 
        v0, 
        method='SLSQP', 
        bounds=bounds,
        constraints=cons_dict,
        options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
    )

    final_xs = res.x[:N]
    final_ys = res.x[N:2*N]
    final_rs = res.x[2*N:3*N]

    # 3. Safety adjustment to guarantee strict validity against numerical drift
    # Find the minimum constraint slack
    min_slack = 1.0
    xs_adj, ys_adj, rs_adj = final_xs.copy(), final_ys.copy(), final_rs.copy()
    
    # Check boundaries
    for i in range(N):
        s1 = xs_adj[i] - rs_adj[i]
        s2 = 1.0 - xs_adj[i] - rs_adj[i]
        s3 = ys_adj[i] - rs_adj[i]
        s4 = 1.0 - ys_adj[i] - rs_adj[i]
        min_slack = min(min_slack, s1, s2, s3, s4)

    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs_adj[i] - xs_adj[j])**2 + (ys_adj[i] - ys_adj[j])**2)
            s = d - rs_adj[i] - rs_adj[j]
            min_slack = min(min_slack, s)

    # If any constraint is slightly negative due to precision, scale down radii minimally
    if min_slack < 0:
        scale = 1.0 + min_slack / np.max(rs_adj)
        if scale < 1.0:
            rs_adj *= scale
            
    centers = np.column_stack([xs_adj, ys_adj])
    total_radius = float(np.sum(rs_adj))
    
    return centers, rs_adj, total_radius
