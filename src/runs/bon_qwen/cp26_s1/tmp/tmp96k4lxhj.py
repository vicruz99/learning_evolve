import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    def objective(params):
        # Objective: minimize negative sum of radii (maximize sum of radii)
        return -np.sum(params[2*n : 4*n])

    def boundary_constraints(params):
        c = params[0:2*n].reshape(n, 2)
        r = params[2*n:4*n]
        constraints = []
        for i in range(n):
            constraints.append(c[i, 0] - r[i])
            constraints.append(1.0 - (c[i, 0] + r[i]))
            constraints.append(c[i, 1] - r[i])
            constraints.append(1.0 - (c[i, 1] + r[i]))
            constraints.append(r[i]) # Non-negative radius
        return np.array(constraints)

    def overlap_constraints(params):
        c = params[0:2*n].reshape(n, 2)
        r = params[2*n:4*n]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(c[i] - c[j])
                constraints.append(dist - (r[i] + r[j]))
        return np.array(constraints)

    def get_bounds():
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
        for _ in range(n):
            bounds.append((0.0, 0.5)) # r
        return bounds

    # Stage 1: Hexagonal Initialization
    rows, cols = 5, 6
    r_init = 0.09
    x_spacing = 2 * r_init
    y_spacing = np.sqrt(3) * r_init
    
    centers = []
    radii = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n: break
            x = (c + 0.5) * x_spacing + (r % 2) * (x_spacing / 2) + 0.05
            y = r * y_spacing + 0.05
            if x < 1.0 and y < 1.0:
                centers.append([x, y])
                radii.append(r_init)
                count += 1
        if count >= n: break
    
    centers = np.array(centers[:n])
    radii = np.array(radii[:n])
    
    # Normalize to unit square roughly
    centers = (centers - centers.min(axis=0)) / (centers.max(axis=0) - centers.min(axis=0)) * 0.8 + 0.1
    radii = np.ones(n) * 0.09

    # Combine into params array: [x0, y0, ..., x25, y25, r0, ..., r25]
    init_params = np.concatenate([centers.flatten(), radii])
    
    # Stage 2: SLSQP Optimization
    bounds = get_bounds()
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]

    for seed in range(5): # Try 5 different perturbations
        # Add noise to initialization
        noise = np.random.randn(len(init_params)) * 0.01
        current_params = init_params + noise
        current_params = np.clip(current_params, 0.0, 1.0) # Ensure basic bounds
        
        res = minimize(
            objective, 
            current_params, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        if res.success:
            final_params = res.x
            curr_centers = final_params[0:2*n].reshape(n, 2)
            curr_radii = final_params[2*n:4*n]
            
            # Validate and calculate sum
            if validate_packing(curr_centers, curr_radii):
                curr_sum = np.sum(curr_radii)
                if curr_sum > best_sum_radii:
                    best_sum_radii = curr_sum
                    best_centers = curr_centers.copy()
                    best_radii = curr_radii.copy()

    # Fallback if optimization fails (should not happen with hex init)
    if best_centers is None:
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii

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