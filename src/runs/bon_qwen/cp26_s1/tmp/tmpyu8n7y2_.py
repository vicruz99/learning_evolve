import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist

def build_constraint_matrix(n):
    """Precompute the static inequality matrix for the radii LP."""
    m = n * (n - 1) // 2
    A = np.zeros((m, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            idx += 1
    return A

def solve_radii(centers, A_ub):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = len(centers)
    dists = pdist(centers)
    b_ub = dists.copy()

    var_bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        var_bounds.append((0.0, ub))

    # Maximize sum(r) <=> Minimize -sum(r)
    c_obj = -np.ones(n)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method='highs')
    
    if res.success:
        return -res.fun, -res.x
    return 0.0, np.zeros(n)

def objective(centers_flat, A_ub):
    """Objective function for center optimization."""
    centers = centers_flat.reshape((26, 2))
    s, _ = solve_radii(centers, A_ub)
    return -s  # Powell minimizes, so we return negative sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    A_ub = build_constraint_matrix(n)

    # 1. Initialize centers with a perturbed uniform grid
    xs = np.linspace(0.15, 0.85, 6)
    ys = np.linspace(0.15, 0.85, 5)
    centers_init = []
    for y in ys:
        for x in xs:
            centers_init.append([x, y])
            if len(centers_init) == n:
                break
        if len(centers_init) == n:
            break
    centers_init = np.array(centers_init)

    # Small random perturbation breaks symmetry and helps escape trivial local minima
    np.random.seed(42)
    centers_init += np.random.uniform(-0.02, 0.02, centers_init.shape)
    centers_init = np.clip(centers_init, 0.05, 0.95)

    # 2. Optimize centers using Powell's method
    # The LP inside 'objective' ensures radii are always optimal for the current centers
    bounds = [(0.0, 1.0) for _ in range(52)]
    res = minimize(objective, centers_init.flatten(), args=(A_ub,), method='Powell',
                   bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})

    best_centers = res.x.reshape((26, 2))
    best_sum, best_radii = solve_radii(best_centers, A_ub)

    # 3. Apply negligible shrink to guarantee strict compliance with validator tolerance
    best_radii *= 0.99999

    return best_centers, best_radii, best_sum