import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(vars[2 * N_CIRCLES:])

def constraint_fun(vars):
    """Inequality constraints: boundary containment and non-overlap"""
    n = N_CIRCLES
    centers = vars[:2 * n].reshape(n, 2)
    radii = vars[2 * n:]
    
    cons = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    cons.append(centers[:, 0] - radii)
    cons.append(1.0 - centers[:, 0] - radii)
    cons.append(centers[:, 1] - radii)
    cons.append(1.0 - centers[:, 1] - radii)
    
    # Non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            d2 = np.sum((centers[i] - centers[j]) ** 2)
            rs = radii[i] + radii[j]
            cons.append(d2 - rs ** 2)
            
    return np.concatenate(cons)

def get_initial_config(n, seed):
    """Generate a perturbed hexagonal lattice initialization"""
    np.random.seed(seed)
    rows = 5
    cols = 6
    r_init = 0.1
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= n:
                break
            y = 0.1 + i * np.sqrt(3) * r_init
            x = 0.1 + j * 2 * r_init + (i % 2) * r_init
            pts.append([x, y])
        if len(pts) >= n:
            break
            
    pts = np.array(pts[:n])
    # Add noise to break symmetry and help escape local minima
    pts += np.random.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.01, 0.99)
    radii = np.full(n, 0.05)
    return pts, radii

def solve_packing():
    """Run multi-start SLSQP optimization to find optimal packing"""
    best_sum_r = -1.0
    best_centers = None
    best_radii = None

    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.2)] * N_CIRCLES
    con = {'type': 'ineq', 'fun': constraint_fun}

    # Multiple starts to ensure global convergence
    for seed in range(5):
        centers, radii = get_initial_config(N_CIRCLES, seed)
        x0 = np.concatenate([centers.flatten(), radii])

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=con,
                           options={'maxiter': 1000, 'ftol': 1e-11, 'disp': False})
            curr_sum = np.sum(res.x[2 * N_CIRCLES:])
            if curr_sum > best_sum_r:
                best_sum_r = curr_sum
                best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
                best_radii = res.x[2 * N_CIRCLES:]
        except Exception:
            continue

    return best_centers, best_radii, best_sum_r

def run_packing():
    """Entry point: solve and return validated packing configuration"""
    centers, radii, total_r = solve_packing()
    # Ensure non-negativity compliance
    radii = np.maximum(radii, 0.0)
    return centers, radii, total_r