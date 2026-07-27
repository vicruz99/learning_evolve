import numpy as np
from scipy.optimize import linprog

def get_lp_radii(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)  # Maximize sum(r) => minimize -sum(r)
    n_pairs = n * (n - 1) // 2
    n_bound = 4 * n
    A_ub = np.zeros((n_pairs + n_bound, n))
    b_ub = np.zeros(n_pairs + n_bound)

    idx = 0
    # Pairwise non-overlap constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1

    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1

    bounds = [(0.0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x
    return 0.0, np.zeros(n)

def run_packing() -> tuple:
    n = 26
    best_sum = -1.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)

    for trial in range(6):
        np.random.seed(trial * 137 + 42)
        # Initialize centers with some spread
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        
        # Pre-spreading phase to avoid initial clustering
        for _ in range(80):
            forces = np.zeros_like(centers)
            for i in range(n):
                for j in range(i + 1, n):
                    v = centers[i] - centers[j]
                    d = np.linalg.norm(v)
                    if d < 0.08 and d > 1e-5:
                        f = (0.08 - d) / (d + 1e-9)
                        forces[i] += v * f
                        forces[j] -= v * f
            centers += forces * 0.005
            centers = np.clip(centers, 0.02, 0.98)

        # Main alternating optimization loop
        radii = np.zeros(n)
        for step in range(200):
            obj, radii = get_lp_radii(centers)
            if obj > best_sum:
                best_sum = obj
                best_centers = centers.copy()
                best_radii = radii.copy()

            # Decay step size for stability
            dt = 0.008 * (0.98 ** (step // 15))
            forces = np.zeros_like(centers)
            
            # Repulsion for active constraints (tight packing)
            for i in range(n):
                for j in range(i + 1, n):
                    v = centers[i] - centers[j]
                    d = np.linalg.norm(v)
                    if d < 1e-6: 
                        continue
                    # If circles are close to touching, push apart
                    if radii[i] + radii[j] > d - 1e-3:
                        strength = 1.0 + (radii[i] + radii[j])
                        forces[i] += strength * v / (d + 1e-9)
                        forces[j] -= strength * v / (d + 1e-9)
            
            # Boundary repulsion to keep circles strictly inside
            for i in range(n):
                r = radii[i]
                if centers[i, 0] - r < 1e-3: forces[i, 0] += 3.0
                elif centers[i, 0] + r > 1.0 - 1e-3: forces[i, 0] -= 3.0
                if centers[i, 1] - r < 1e-3: forces[i, 1] += 3.0
                elif centers[i, 1] + r > 1.0 - 1e-3: forces[i, 1] -= 3.0

            centers += dt * forces
            centers = np.clip(centers, 1e-4, 1.0 - 1e-4)

        # Final evaluation
        obj, radii = get_lp_radii(centers)
        if obj > best_sum:
            best_sum = obj
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum