import numpy as np
import scipy.optimize as opt

def get_coords(r, u, v):
    """Converts (r, u, v) parameters to (x, y) center coordinates."""
    x = r + (1 - 2 * r) * u
    y = r + (1 - 2 * r) * v
    return np.column_stack([x, y])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    best_sum = 0.0
    best_params = None

    # Objective function: minimize negative sum of radii
    def objective(params):
        radii = params[0::3]
        return -np.sum(radii)

    # Constraint function: distance squared >= (r_i + r_j)^2
    def constraint(params):
        radii = params[0::3]
        us = params[1::3]
        vs = params[2::3]
        centers = get_coords(radii, us, vs)
        
        constraints = []
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = np.sum((centers[i] - centers[j]) ** 2)
                min_dist_sq = (radii[i] + radii[j]) ** 2
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)

    # Bounds for variables: r in [0, 0.5], u in [0, 1], v in [0, 1]
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 0.5), (0.0, 1.0), (0.0, 1.0)])

    # Constraints setup for SLSQP
    cons = {'type': 'ineq', 'fun': constraint}

    # Run multiple optimizations with different initializations
    for seed in range(10):
        np.random.seed(seed)
        
        # Initialize with a perturbed grid-like arrangement
        r_init = np.full(N, 0.09)
        u_init = np.random.rand(N)
        v_init = np.random.rand(N)
        
        # Create a structured initial grid for better convergence
        # 5x5 grid covers 25 circles
        grid_size = 5
        points = np.linspace(0.1, 0.9, grid_size)
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < N:
                    u_init[count] = points[i]
                    v_init[count] = points[j]
                    count += 1
        
        # Random perturbation
        u_init += np.random.uniform(-0.1, 0.1, N)
        v_init += np.random.uniform(-0.1, 0.1, N)
        u_init = np.clip(u_init, 0, 1)
        v_init = np.clip(v_init, 0, 1)

        params_init = np.array([r_init, u_init, v_init]).T.flatten()

        try:
            res = opt.minimize(objective, params_init, method='SLSQP', 
                               bounds=bounds, constraints=cons, 
                               options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or -res.fun > best_sum:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_params = res.x.copy()
        except Exception:
            continue

    # Extract final results
    radii = best_params[0::3]
    us = best_params[1::3]
    vs = best_params[2::3]
    centers = get_coords(radii, us, vs)
    
    return centers, radii, float(np.sum(radii))

if __name__ == "__main__":
    centers, radii, total_r = run_packing()
    print(f"Sum of radii: {total_r}")
    print(f"Max radius: {np.max(radii)}")
    print(f"Min radius: {np.min(radii)}")