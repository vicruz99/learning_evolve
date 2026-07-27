import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(v):
    """Vectorized constraint evaluation for SLSQP."""
    cs = v.reshape(-1, 3)
    c = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c.append(cs[:, 0] - cs[:, 2])
    c.append(1.0 - cs[:, 0] - cs[:, 2])
    c.append(cs[:, 1] - cs[:, 2])
    c.append(1.0 - cs[:, 1] - cs[:, 2])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    diffs = cs[:, :2][:, np.newaxis, :] - cs[:, :2][np.newaxis, :, :]
    dist_sq = np.sum(diffs**2, axis=2)
    r_sum = cs[:, 2][:, np.newaxis] + cs[:, 2][np.newaxis, :]
    
    idx = np.triu_indices(N, k=1)
    c.append(dist_sq[idx] - r_sum[idx]**2)
    
    return np.concatenate(c)

def objective_func(v):
    """Negative sum of radii (to be minimized)."""
    return -np.sum(v[2::3])

def run_optimizer(v0):
    """Run SLSQP optimization from initial vector v0."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    constraints = {'type': 'ineq', 'fun': compute_constraints}
    options = {'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
    
    try:
        res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options=options)
        if not np.isnan(res.fun):
            return res.x
    except Exception:
        pass
    return v0

def generate_hex_start():
    """Generate a dense hexagonal lattice starting configuration."""
    r_start = 0.085
    d_start = 0.16
    centers = []
    row_counts = [5, 5, 5, 5, 6]
    
    for i, count in enumerate(row_counts):
        y = r_start + i * (np.sqrt(3)/2 * d_start)
        x_start = r_start + d_start/2 if i % 2 == 1 else r_start
        for j in range(count):
            centers.append([x_start + j * d_start, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
            
    return np.array(centers[:N])

def generate_grid_start():
    """Generate a perturbed grid starting configuration."""
    np.random.seed(42)
    r_start = 0.05
    xs = np.linspace(r_start, 1.0 - r_start, 6)
    ys = np.linspace(r_start, 1.0 - r_start, 5)
    
    centers = []
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
            
    centers = np.array(centers[:N])
    centers += np.random.uniform(-0.02, 0.02, size=(N, 2))
    return np.clip(centers, r_start, 1.0 - r_start)

def run_packing():
    # Generate initial configurations
    initial_configs = [generate_hex_start(), generate_grid_start()]
    
    best_v = None
    best_sum = -1.0
    
    # Optimize from each start
    for centers_init in initial_configs:
        radii_init = np.full(N, 0.02)
        v0 = np.concatenate([centers_init.ravel(), radii_init])
        v_opt = run_optimizer(v0)
        
        current_sum = np.sum(v_opt[2*N:])
        if current_sum > best_sum:
            best_sum = current_sum
            best_v = v_opt.copy()
            
    # Iterative perturbation refinement to escape local optima
    for _ in range(4):
        v_perturbed = best_v.copy()
        # Add small noise to centers only
        noise = np.random.uniform(-0.001, 0.001, size=(2 * N))
        v_perturbed[:2*N] += noise
        v_perturbed = np.clip(v_perturbed[:2*N], 0.0, 1.0)
        
        v_opt = run_optimizer(v_perturbed)
        current_sum = np.sum(v_opt[2*N:])
        if current_sum > best_sum:
            best_sum = current_sum
            best_v = v_opt.copy()
            
    # Extract and finalize
    centers_opt = best_v[:2*N].reshape(N, 2)
    radii_opt = best_v[2*N:]
    
    # Ensure strict feasibility for validation (handles numerical slack)
    centers_opt = np.clip(centers_opt, 0.0, 1.0)
    radii_opt = np.clip(radii_opt, 1e-9, 0.5)
    
    # Quick feasibility correction if needed
    c_vals = compute_constraints(best_v)
    min_c = np.min(c_vals)
    if min_c < -1e-8:
        # Scale radii down proportionally to fix violations
        radii_opt *= (1.0 + min_c)
        # Ensure centers stay within bounds after radius change
        centers_opt[:, 0] = np.clip(centers_opt[:, 0], radii_opt, 1.0 - radii_opt)
        centers_opt[:, 1] = np.clip(centers_opt[:, 1], radii_opt, 1.0 - radii_opt)
        
    return centers_opt, radii_opt, np.sum(radii_opt)