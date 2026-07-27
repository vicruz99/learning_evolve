import numpy as np
from scipy.optimize import linprog, minimize

def compute_optimal_radii(centers):
    """
    Given fixed centers, solve LP to find radii that maximize sum(r_i)
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    
    # Compute pairwise Euclidean distances efficiently using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Compute minimum distance to square boundaries for each center
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Linear Programming Setup
    # Objective: Maximize sum(r_i)  <==>  Minimize -sum(r_i)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Constraints: r_i + r_j <= dist(c_i, c_j)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Constraints: r_i <= limit_i (boundary clearance)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(limits[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def objective_function(x_flat):
    """Objective for the center optimizer: negative of max sum of radii."""
    n = 26
    centers = x_flat.reshape(-1, 2)
    # Keep centers strictly inside [0,1] to prevent degenerate LP cases
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    sum_r, _ = compute_optimal_radii(centers)
    return -sum_r

def run_packing():
    n = 26
    best_val = 0.0
    best_centers = None
    
    def create_hex_grid():
        """Generate a dense hexagonal lattice as a high-quality initial guess."""
        s = 0.19  # Spacing parameter
        pts = []
        y = s / 2
        row_idx = 0
        while y + s/2 <= 1.0:
            offset = (row_idx % 2) * s / 2
            x = offset
            while x + s/2 <= 1.0:
                pts.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row_idx += 1
            
        if len(pts) < n:
            pts = [list(np.random.rand(2)) for _ in range(n)]
        else:
            pts = pts[:n]
        return np.array(pts)

    # Multiple restarts with perturbed hex grids to escape local minima
    for restart in range(8):
        if restart == 0:
            x0 = create_hex_grid().flatten()
        else:
            x0 = create_hex_grid().flatten() + np.random.randn(2*n) * 0.03
            x0 = np.clip(x0, 0.02, 0.98)
            
        try:
            # Nelder-Mead is robust for non-smooth objectives like this
            res = minimize(objective_function, x0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-9})
            curr_val = -res.fun
            if curr_val > best_val:
                best_val = curr_val
                best_centers = np.clip(res.x.reshape(-1, 2), 0.0, 1.0)
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = create_hex_grid()
        
    # Final precise LP solve for the best centers
    final_sum, final_radii = compute_optimal_radii(best_centers)
    
    return best_centers, final_radii, final_sum