import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the packing problem for 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    best_solution = None
    best_score = -np.inf

    def objective(vars_arr):
        # vars_arr contains [x1, y1, r1, x2, y2, r2, ...]
        radii = vars_arr[2::3]
        return -np.sum(radii)

    def constraint_non_overlap(vars_arr, i, j):
        x1, y1, r1 = vars_arr[3*i:3*i+3]
        x2, y2, r2 = vars_arr[3*j:3*j+3]
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        min_dist_sq = (r1 + r2)**2
        # We want dist_sq >= min_dist_sq, so dist_sq - min_dist_sq >= 0
        return dist_sq - min_dist_sq

    def constraint_boundary_left(vars_arr, i):
        x, r = vars_arr[3*i], vars_arr[3*i+2]
        return x - r

    def constraint_boundary_right(vars_arr, i):
        x, r = vars_arr[3*i], vars_arr[3*i+2]
        return (1 - x) - r

    def constraint_boundary_bottom(vars_arr, i):
        y, r = vars_arr[3*i+1], vars_arr[3*i+2]
        return y - r

    def constraint_boundary_top(vars_arr, i):
        y, r = vars_arr[3*i+1], vars_arr[3*i+2]
        return (1 - y) - r

    def constraint_radius_positive(vars_arr, i):
        return vars_arr[3*i+2]

    # Hexagonal lattice initialization
    def get_hex_initialization(n, seed=0):
        rng = np.random.default_rng(seed)
        # Estimate radius for hex packing of N circles
        # Area ~ N * pi * r^2 ~ 0.9 -> r ~ sqrt(0.9 / (N*pi))
        r_init = np.sqrt(0.9 / (n * np.pi)) * 0.95 # slightly smaller to start
        
        # Try to fit in a grid pattern first then perturb
        # Hexagonal spacing
        dx = 2 * r_init
        dy = np.sqrt(3) * r_init
        
        rows = []
        count = 0
        y = r_init
        while count < n:
            # Determine x positions for this row
            # Start at r_init
            row_centers = []
            x = r_init
            while x + r_init <= 1.0:
                row_centers.append(x)
                x += dx
                count += 1
                if count >= n:
                    break
            rows.append(row_centers)
            y += dy
            # Shift next row
            r_init_temp = r_init # Use same radius
            # Adjust start x for shift if needed, but simple grid is okay for init
            
        # Flatten to centers
        centers = []
        for r_idx, row in enumerate(rows):
            for x in row:
                # Shift odd rows for hexagonal packing
                offset = r_init if r_idx % 2 == 1 else 0
                cx = x + offset
                cy = r_idx * dy + r_init
                centers.append([cx, cy])
                
        centers = np.array(centers[:n])
        
        # Add small random perturbation to break symmetry
        centers += rng.uniform(-0.005, 0.005, size=centers.shape)
        
        # Ensure boundaries
        centers = np.clip(centers, r_init, 1 - r_init)
        
        vars_arr = np.zeros(n * 3)
        vars_arr[::3] = centers[:, 0]
        vars_arr[1::3] = centers[:, 1]
        vars_arr[2::3] = r_init
        
        return vars_arr

    # Optimization loop with multiple restarts
    for seed in range(10):
        try:
            x0 = get_hex_initialization(n_circles, seed=seed)
            
            constraints = []
            # Add overlap constraints
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: constraint_non_overlap(v, i, j)
                    })
            # Add boundary constraints
            for i in range(n_circles):
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_left(v, i)})
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_right(v, i)})
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_bottom(v, i)})
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_top(v, i)})
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_radius_positive(v, i)})

            # Bounds for variables
            bounds = []
            for i in range(n_circles):
                bounds.extend([
                    (0, 1), # x
                    (0, 1), # y
                    (0, 0.5) # r
                ])

            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 2000, 'disp': False}
            )

            if res.success:
                current_sum = -res.fun
                if current_sum > best_score:
                    best_score = current_sum
                    best_solution = res.x

        except Exception as e:
            print(f"Optimization failed with seed {seed}: {e}")
            continue

    if best_solution is None:
        # Fallback to a simple valid packing if optimization fails
        # 5x5 grid with 1 extra, small radius
        r = 0.05
        centers = []
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + j * 0.2, 0.1 + i * 0.2])
        centers.append([0.5, 0.5]) # Center
        centers = np.array(centers[:26])
        radii = np.full(26, r)
        return centers, radii, np.sum(radii)

    # Extract results
    centers = np.column_stack((best_solution[::3], best_solution[1::3]))
    radii = best_solution[2::3]
    
    # Post-processing to ensure strict validity (clip tiny violations)
    # This is a safeguard against numerical precision issues
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        
        # Boundary clipping
        if x - r < 0: centers[i, 0] = r
        if x + r > 1: centers[i, 0] = 1 - r
        if y - r < 0: centers[i, 1] = r
        if y + r > 1: centers[i, 1] = 1 - r
        
    return centers, radii, float(np.sum(radii))

if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    if validate_packing(centers, radii):
        print(f"Valid packing found with sum of radii: {s_r}")
    else:
        print("Invalid packing generated!")