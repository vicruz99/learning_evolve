import numpy as np
from scipy.optimize import differential_evolution

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
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

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Initialization: Hexagonal grid pattern
    def init_hex_grid():
        centers = []
        # Approximate radius for 26 circles is around 0.1
        r_est = 0.101
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        x, y = 0.5, 0.5
        
        # Spiral out from center to place 26 points
        # Simple grid approach is more robust for N=26
        # 5 columns, 6 rows (staggered)
        rows = 6
        cols = 5
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                # Stagger every other row
                x_offset = (i % 2) * (dx / 2)
                # Scale to fit in 0-1
                # Adjust bounds
                cx = 0.1 + j * dx + x_offset
                cy = 0.1 + i * dy
                if cx < 1 - r_est and cy < 1 - r_est:
                    centers.append([cx, cy])
            if len(centers) >= n:
                break
        return np.array(centers[:n])

    # Optimization wrapper for Differential Evolution
    def objective(vars):
        centers = vars[:n*2].reshape((n, 2))
        radii = vars[2*n:]
        
        # Apply bounds strictly for penalty
        penalty = 0.0
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Boundary penalties
            if x < r: penalty += (x - r)**2 * 1000
            if x > 1 - r: penalty += (x - (1 - r))**2 * 1000
            if y < r: penalty += (y - r)**2 * 1000
            if y > 1 - r: penalty += (y - (1 - r))**2 * 1000
            
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2 * 1000
        
        return penalty - np.sum(radii) # Maximize sum (minimize negative sum)

    # Bounds for optimization: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * (n * 2) + [(0, 0.5)] * n
    
    # Run Differential Evolution for refinement
    result = differential_evolution(objective, bounds, seed=42, 
                                    popsize=15, maxiter=1000, tol=1e-6, 
                                    polish=True)
    
    if result.success:
        vars_opt = result.x
        centers = vars_opt[:n*2].reshape((n, 2))
        radii = vars_opt[2*n:]
        
        # Clamp and validate
        valid = validate_packing(centers, radii)
        if valid:
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers
                best_radii = radii

    return best_centers, best_radii, best_sum