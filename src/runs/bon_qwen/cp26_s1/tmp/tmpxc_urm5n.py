import numpy as np
from scipy.optimize import minimize
import math

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

def compute_overlap_penalty(centers, radii):
    """
    Computes a penalty for overlapping circles.
    Returns 0 if no overlaps, positive value otherwise.
    """
    n = len(radii)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                violation = min_dist - dist
                penalty += violation**2
    return penalty

def compute_boundary_penalty(centers, radii):
    """
    Computes a penalty for circles going outside the unit square.
    """
    penalty = 0.0
    for i in range(len(radii)):
        x, y = centers[i]
        r = radii[i]
        
        # Check boundaries
        if x - r < 0: penalty += (x - r)**2
        if x + r > 1: penalty += (x + r - 1)**2
        if y - r < 0: penalty += (y - r)**2
        if y + r > 1: penalty += (y + r - 1)**2
        
    return penalty

def objective_function(params, n_circles):
    """
    Objective function to minimize: -(Sum of radii) + Penalties
    params: flat array of [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        idx = i * 3
        centers[i] = params[idx:idx+2]
        radii[i] = params[idx+2]
        
    # Ensure radii are non-negative (clamping inside objective is a bit hacky but helps optimization)
    # Ideally handled by bounds, but let's be safe.
    # Actually, bounds in scipy are better.
    
    # Sum of radii (we want to maximize this, so minimize negative)
    sum_radii = np.sum(radii)
    
    # Penalties
    overlap_pen = compute_overlap_penalty(centers, radii)
    boundary_pen = compute_boundary_penalty(centers, radii)
    
    # Weight for penalties. High weight ensures constraints are respected.
    # We want to maximize sum_radii, so we subtract penalties from the "score" we want to maximize?
    # No, we minimize: -sum_radii + lambda * penalties
    penalty_weight = 100.0
    
    return -sum_radii + penalty_weight * (overlap_pen + boundary_pen)

def get_initial_guess(n_circles, grid_spacing=0.15):
    """
    Generates initial guesses based on a hexagonal lattice.
    """
    points = []
    # Hexagonal grid parameters
    # Row spacing: spacing * sqrt(3)/2
    # Col spacing: spacing
    
    # Generate enough points
    y = 0
    row = 0
    while y <= 1.0:
        x = 0
        # Shift every other row
        if row % 2 == 1:
            x = grid_spacing / 2.0
        
        while x <= 1.0:
            points.append([x, y])
            x += grid_spacing
        y += grid_spacing * math.sqrt(3) / 2.0
        row += 1
    
    # Shuffle points to randomize selection slightly or pick best
    np.random.shuffle(points)
    
    # Select n_circles points
    selected_points = points[:n_circles]
    
    # Initial radii: small positive number
    init_radii = np.full(n_circles, 0.05)
    
    # Construct params
    params = []
    for i in range(n_circles):
        params.extend(selected_points[i])
        params.append(init_radii[i])
        
    return np.array(params)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple restarts to find global optimum
    n_restarts = 10
    
    for restart in range(n_restarts):
        # Generate initial guess
        # Use a slightly random spacing or shift
        shift_x = np.random.random() * 0.1
        shift_y = np.random.random() * 0.1
        spacing = 0.12 + np.random.random() * 0.05
        
        # Generate grid points
        points = []
        y = shift_y
        row = 0
        while y <= 1.0:
            x = shift_x
            if row % 2 == 1:
                x = shift_x + spacing / 2.0
            
            while x <= 1.0:
                # Check if point is inside square (with margin for radius)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y])
                x += spacing
            y += spacing * math.sqrt(3) / 2.0
            row += 1
        
        # If not enough points, fall back to random or dense grid
        if len(points) < n_circles:
            # Dense random
            points = np.random.uniform(0, 1, size=(n_circles, 2))
        else:
            # Randomly select n_circles
            indices = np.random.choice(len(points), n_circles, replace=False)
            points = [points[i] for i in indices]
            
        # Initial params: centers + radii
        # Start with small radii to avoid immediate heavy penalties
        params = []
        for p in points:
            params.extend(p)
            params.append(0.08) # Initial radius guess
            
        params = np.array(params)
        
        # Bounds: x, y in [0, 1], r in [0, 0.5] (max possible radius in unit square is 0.5)
        bounds = []
        for i in range(n_circles):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])
            
        # Optimization
        # Use L-BFGS-B
        res = minimize(objective_function, params, args=(n_circles,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-6})
        
        if res.success or res.fun < -2.0: # Heuristic for success
            # Extract results
            centers = np.zeros((n_circles, 2))
            radii = np.zeros(n_circles)
            for i in range(n_circles):
                idx = i * 3
                centers[i] = res.x[idx:idx+2]
                radii[i] = res.x[idx+2]
                
            # Validate and clean up
            # Clamp coordinates and radii
            centers = np.clip(centers, 0, 1)
            radii = np.clip(radii, 0, 0.5)
            
            # Check validity
            if validate_packing(centers, radii):
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()

    # If no valid packing found (unlikely), return a safe grid
    if best_centers is None:
        # Fallback: 5x5 grid + 1 small
        best_centers = np.zeros((26, 2))
        best_radii = np.zeros(26)
        idx = 0
        for i in range(5):
            for j in range(5):
                best_centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                best_radii[idx] = 0.1
                idx += 1
        # 26th circle
        best_centers[25] = [0.5, 0.5]
        best_radii[25] = 0.01 # Tiny
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)