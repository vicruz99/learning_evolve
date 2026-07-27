import numpy as np
import scipy.optimize as opt

def get_boundaries(centers):
    """Compute distance from each center to the nearest boundary of the unit square."""
    x = centers[:, 0]
    y = centers[:, 1]
    return np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

def solve_packing_lp(centers):
    """
    Solve the Linear Program to find optimal radii for fixed centers.
    Maximizes sum(r_i) subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    if n == 0:
        return np.array([]), 0.0

    # Pairwise distances between centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))

    # LP Objective: Maximize sum(r_i)  <=>  Minimize -sum(r_i)
    c_obj = -np.ones(n)

    # Constraints: r_i + r_j <= d_ij  and  r_i <= boundary_i
    # Construct A_ub and b_ub
    n_pairs = n * (n - 1) // 2
    n_bounds = n
    A_ub = np.zeros((n_pairs + n_bounds, n))
    b_ub = np.zeros(n_pairs + n_bounds)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d[i, j]
            idx += 1

    # Boundary constraints
    b_val = get_boundaries(centers)
    A_ub[idx:, :] = np.eye(n)
    b_ub[idx:] = b_val

    bounds = [(0, None)] * n

    try:
        # Use HiGHS solver for efficiency
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
        else:
            # Fallback if LP fails
            return np.full(n, 1e-9), 0.0
    except Exception:
        return np.full(n, 1e-9), 0.0

def evaluate(centers_flat):
    """Objective function for the outer optimization."""
    centers = centers_flat.reshape(-1, 2)
    centers = np.clip(centers, 0, 1)
    radii, total = solve_packing_lp(centers)
    return -total  # Minimize negative sum

def generate_hex_grid(n):
    """Generate an initial hexagonal grid of n points in [0,1]^2."""
    cols = int(np.ceil(np.sqrt(n * np.sqrt(3) / 2))) + 1
    rows = int(np.ceil(n / cols)) + 1
    
    pts = []
    dy = 1.0 / rows
    dx = 1.0 / cols
    
    for r in range(rows):
        for c in range(cols):
            if len(pts) >= n:
                break
            x = (c + 0.5) * dx
            y = (r + 0.5 + (c % 2) * 0.5) * dy
            pts.append([x, y])
        if len(pts) >= n:
            break
            
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    initial_centers = generate_hex_grid(n)
    
    # Optimize centers using Powell method which handles many variables well
    best_res = None
    best_val = np.inf
    
    # Run optimization from the hexagonal start
    # Adding a small random perturbation to break symmetries if needed
    x0 = initial_centers.flatten()
    
    res = opt.minimize(evaluate, x0, method='Powell', 
                       options={'maxiter': 1000, 'ftol': 1e-6, 'xtol': 1e-6})
    
    if res.fun < best_val:
        best_val = res.fun
        best_res = res
        
    # Fallback to Nelder-Mead if Powell didn't improve or failed
    if best_res is None or best_val > -2.0: # Heuristic check
        res2 = opt.minimize(evaluate, x0, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res2.fun < best_val:
            best_val = res2.fun
            best_res = res2
            
    centers = best_res.x.reshape(-1, 2)
    radii, _ = solve_packing_lp(centers)
    
    return centers, radii, np.sum(radii)