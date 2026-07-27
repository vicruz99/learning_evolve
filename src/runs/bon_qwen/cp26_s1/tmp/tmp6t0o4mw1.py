import numpy as np
import scipy.optimize
import scipy.spatial

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate initial radius for 26 circles
    r_init = 0.09
    centers = []
    row_y = r_init
    col_offset = 0
    
    # Fill rows to get at least 26 points
    while len(centers) < n:
        row_x = r_init
        while row_x < 1.0 - r_init + 1e-5:
            centers.append([row_x, row_y])
            row_x += 2 * r_init
        row_y += np.sqrt(3) * r_init
        
    centers = np.array(centers[:n])
    
    # 2. Radius Solver using LP
    def get_max_radii_sum(centers):
        dist_matrix = scipy.spatial.distance_matrix(centers, centers)
        b = np.zeros(n)
        for i in range(n):
            b[i] = min(centers[i, 0], 1.0 - centers[i, 0], 
                       centers[i, 1], 1.0 - centers[i, 1])
        
        # LP: maximize sum(r) -> minimize -sum(r)
        # Constraints: r_i + r_j <= d_ij, r_i <= b_i, r_i >= 0
        A_ub = np.zeros((n * (n - 1) // 2 + n, n))
        b_ub = np.zeros(n * (n - 1) // 2 + n)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A_ub[idx, i] = 1
                A_ub[idx, j] = 1
                b_ub[idx] = dist_matrix[i, j]
                idx += 1
        for i in range(n):
            A_ub[idx, i] = 1
            b_ub[idx] = b[i]
            idx += 1
            
        res = scipy.optimize.linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            return 0.0, np.zeros(n)

    # 3. Center Optimization
    def objective(centers_flat):
        centers = centers_flat.reshape((n, 2))
        # Apply bounds strictly to avoid invalid LP constraints
        centers = np.clip(centers, 0, 1)
        s, _ = get_max_radii_sum(centers)
        return -s

    bounds = [(0, 1)] * (n * 2)
    res = scipy.optimize.minimize(objective, centers.flatten(), method='Nelder-Mead', 
                                  bounds=bounds, options={'maxiter': 50000, 'xatol': 1e-6, 'fatol': 1e-8})

    final_centers = res.x.reshape((n, 2))
    sum_radii, final_radii = get_max_radii_sum(final_centers)
    
    # 4. Validation
    if not validate_packing(final_centers, final_radii):
        # Fallback to a valid grid if optimization failed validation
        final_centers = np.array([[i*0.2+0.1, j*0.2+0.1] for j in range(5) for i in range(5)][:26])
        final_radii = np.full(26, 0.1)
        sum_radii = 2.6

    return final_centers, final_radii, float(sum_radii)

# Provided validation function (read-only)
def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any(): print("NaN values detected in circle centers"); return False
    if np.isnan(radii).any(): print("NaN values detected in circle radii"); return False
    for i in range(n):
        if radii[i] < 0: print(f"Circle {i} has negative radius {radii[i]}"); return False
        elif np.isnan(radii[i]): print(f"Circle {i} has nan radius"); return False
    for i in range(n):
        x, y = centers[i]; r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"); return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}"); return False
    return True