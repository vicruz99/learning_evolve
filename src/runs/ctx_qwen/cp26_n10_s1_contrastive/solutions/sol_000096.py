# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000064 (state 39c4bccd) state=239b6645 sum of radii=2.400000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def obj_de(centers_flat):
    """Objective for Differential Evolution: maximize sum of achievable radii."""
    centers = centers_flat.reshape(N, 2)
    # Distance to boundaries
    dw = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Max feasible radius is limited by walls and nearest neighbor
    r = np.minimum(dw, np.min(dists, axis=1) / 2.0)
    return -np.sum(r)

def constraints_slsqp(vars_vec):
    """Inequality constraints for SLSQP: g(x) >= 0."""
    cx = vars_vec[0::3]
    cy = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c = [cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r]
    
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist2 = dx**2 + dy**2
    
    rs = r[:, None] + r[None, :]
    c.append(dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2)
    return np.concatenate(c)

def obj_slsqp(vars_vec):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def get_max_radii(centers):
    """Compute optimal radii for a fixed center configuration."""
    dw = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.minimum(dw, np.min(dists, axis=1) / 2.0)

def generate_hex_init(seed):
    """Generate a hexagonal lattice initialization with jitter."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    r_est = 0.095
    y = r_est
    row = 0
    idx = 0
    while idx < N and y < 1.0 - r_est + 0.01:
        x_start = r_est if row % 2 == 0 else 2.0 * r_est
        x = x_start
        while idx < N and x < 1.0 - r_est + 0.01:
            centers[idx] = [x, y]
            idx += 1
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    centers += rng.uniform(-0.03, 0.03, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing():
    np.random.seed(42)
    
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    de_bounds = [(0.05, 0.95)] * (2 * N)
    bounds_sl = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_sl = {'type': 'ineq', 'fun': constraints_slsqp}
    
    # Phase 1: Differential Evolution on centers to find optimal topology
    try:
        res_de = differential_evolution(obj_de, de_bounds, popsize=20, maxiter=500,
                                        mutation=(0.5, 1.0), recombination=0.9, seed=42,
                                        polishing=False, init='latinhypercube')
        centers_de = res_de.x.reshape(N, 2)
        r_de = get_max_radii(centers_de)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = centers_de[:, 0]
        x0[1::3] = centers_de[:, 1]
        x0[2::3] = r_de * 0.995  # Shrink slightly for strict interior start
        
        res_sl = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds_sl,
                          constraints=cons_sl, options={'maxiter': 5000, 'ftol': 1e-13})
        if np.min(constraints_slsqp(res_sl.x)) >= -1e-8:
            curr_sum = -res_sl.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = np.column_stack((res_sl.x[0::3], res_sl.x[1::3]))
                best_radii = res_sl.x[2::3]
    except Exception:
        pass
        
    # Phase 2: Structured hexagonal starts to cover alternative basins
    for s in range(5):
        centers_hex = generate_hex_init(s)
        r_hex = get_max_radii(centers_hex)
        x0 = np.zeros(3 * N)
        x0[0::3] = centers_hex[:, 0]
        x0[1::3] = centers_hex[:, 1]
        x0[2::3] = r_hex * 0.995
        
        try:
            res = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds_sl,
                           constraints=cons_sl, options={'maxiter': 4000, 'ftol': 1e-12})
            if np.min(constraints_slsqp(res.x)) >= -1e-8:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    best_radii = res.x[2::3]
        except Exception:
            pass
            
    # Phase 3: Local perturbation refinement on the best found configuration
    if best_centers is not None:
        for _ in range(15):
            x_pert = np.zeros(3 * N)
            x_pert[0::3] = np.clip(best_centers[:, 0] + np.random.randn(N) * 0.005, 0.01, 0.99)
            x_pert[1::3] = np.clip(best_centers[:, 1] + np.random.randn(N) * 0.005, 0.01, 0.99)
            x_pert[2::3] = np.clip(best_radii + np.random.randn(N) * 0.002, 1e-5, 0.49)
            
            try:
                res = minimize(obj_slsqp, x_pert, method='SLSQP', bounds=bounds_sl,
                               constraints=cons_sl, options={'maxiter': 3000, 'ftol': 1e-12})
                if np.min(constraints_slsqp(res.x)) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_radii = res.x[2::3]
            except Exception:
                pass
                
    # Fallback configuration
    if best_centers is None:
        centers_fb = np.zeros((N, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N:
                    centers_fb[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                    idx += 1
        if idx < N:
            centers_fb[idx] = [0.5, 0.5]
        radii_fb = get_max_radii(centers_fb)
        return centers_fb, radii_fb, float(np.sum(radii_fb))
        
    return best_centers, best_radii, float(best_sum)
