# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state bbbe9bd5) state=ed7c023d sum of radii=2.588346 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def compute_lp_radii(centers):
    """Solves the LP to find radii that maximize sum(r_i) for fixed centers."""
    c = -np.ones(N_CIRCLES)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x, 1-x, y, 1-y
    for i in range(N_CIRCLES):
        x, y = centers[i]
        bounds_vals = [x, 1.0 - x, y, 1.0 - y]
        for bv in bounds_vals:
            row = np.zeros(N_CIRCLES)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bv)
            
    # Pairwise constraints: r_i + r_j <= dist_ij
    dx = centers[:, 0, np.newaxis] - centers[:, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            row = np.zeros(N_CIRCLES)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * N_CIRCLES
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(N_CIRCLES, 0.04)

def force_layout_init():
    """Generates a well-spread initial configuration using force simulation."""
    np.random.seed(42)
    centers = np.random.rand(N_CIRCLES, 2) * 0.8 + 0.1
    r_ref = 0.05
    dt = 0.05
    
    # Repulsion phase
    for step in range(1500):
        forces = np.zeros_like(centers)
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d < 1e-5:
                    d = 1e-5
                    diff = np.random.randn(2) * 1e-5
                target = 2.0 * r_ref * 1.15
                if d < target:
                    f_mag = (target - d) / target
                    f_vec = (diff / d) * f_mag
                    forces[i] += f_vec
                    forces[j] -= f_vec
                    
        # Wall confinement
        margin = 0.05
        for i in range(N_CIRCLES):
            if centers[i, 0] < margin: forces[i, 0] += margin - centers[i, 0]
            elif centers[i, 0] > 1.0 - margin: forces[i, 0] -= centers[i, 0] - (1.0 - margin)
            if centers[i, 1] < margin: forces[i, 1] += margin - centers[i, 1]
            elif centers[i, 1] > 1.0 - margin: forces[i, 1] -= centers[i, 1] - (1.0 - margin)
            
        centers += forces * dt
        centers = np.clip(centers, 0.01, 0.99)
        
    return centers

def joint_objective(vars):
    """Objective: maximize sum of radii."""
    r = vars[0::3]
    return -np.sum(r)

def joint_constraint(vars):
    """Constraint: pairwise non-overlap in transformed coordinates."""
    r = vars[0::3]
    u = vars[1::3]
    v = vars[2::3]
    
    # Automatic boundary satisfaction via transformation
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    i, j = np.triu_indices(N_CIRCLES, k=1)
    return dist_sq[i, j] - r_sum_sq[i, j]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initial layout and LP radii
    centers_init = force_layout_init()
    radii_init = compute_lp_radii(centers_init)
    
    # Transform to optimization variables [r, u, v, r, u, v, ...]
    r0 = radii_init * 0.995  # Slight shrink for strict feasibility
    u0 = (centers_init[:, 0] - r0) / (1.0 - 2.0 * r0)
    v0 = (centers_init[:, 1] - r0) / (1.0 - 2.0 * r0)
    u0 = np.clip(u0, 1e-6, 1.0 - 1e-6)
    v0 = np.clip(v0, 1e-6, 1.0 - 1e-6)
    
    vars0 = np.zeros(3 * N_CIRCLES)
    vars0[0::3] = r0
    vars0[1::3] = u0
    vars0[2::3] = v0
    
    bounds = [(1e-4, 0.49), (0, 1), (0, 1)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': joint_constraint}
    
    best_vars = vars0
    best_sum = np.sum(r0)
    
    # 2. SLSQP refinement with multiple perturbations
    for attempt in range(5):
        current_vars = best_vars + np.random.randn(3 * N_CIRCLES) * 0.008
        current_vars = np.clip(current_vars, 1e-4, 0.999) # rough bounds clip
        
        res = minimize(joint_objective, current_vars, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
        
        curr_sum = -res.fun
        if curr_sum > best_sum:
            best_vars = res.x
            best_sum = curr_sum
            
    # 3. Extract optimized centers
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers_opt = np.column_stack((x_opt, y_opt))
    
    # 4. Final LP polish: maximize radii for the fixed optimized centers
    # This guarantees we extract every bit of feasible radius without moving centers
    radii_final = compute_lp_radii(centers_opt)
    
    return centers_opt, radii_final, np.sum(radii_final)
