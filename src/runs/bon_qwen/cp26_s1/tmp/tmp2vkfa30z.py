import numpy as np
import math
from scipy.optimize import minimize, differential_evolution

def get_distance(c1, c2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((c1 - c2) ** 2))

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative
    for i in range(n):
        if radii[i] < 0 or np.isnan(radii[i]):
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Strategy 1: Hexagonal Lattice Initialization with Expansion
    # This aims to find a dense equal-radius packing first
    def hex_packing():
        # Estimate radius for 26 circles (approx 0.1)
        r = 0.1
        # Hexagonal lattice parameters
        # dx = 2r, dy = r*sqrt(3)
        # We try to fit as many as possible in a shifted grid
        
        centers = np.zeros((n, 2))
        radii = np.full(n, r)
        
        # Fill rows
        row_idx = 0
        x, y = r, r
        while row_idx < n:
            # Row shift
            shift = r * (1 if (int((y - r) / (r * math.sqrt(3))) % 2 == 0) else 0)
            x = r + shift
            while x <= 1 - r and row_idx < n:
                centers[row_idx] = [x, y]
                row_idx += 1
                x += 2 * r
            y += r * math.sqrt(3)
            
        # Trim to n if we generated more (though loop condition handles it)
        if row_idx > n:
            centers = centers[:n]
            radii = radii[:n]
            
        return centers, radii

    # Strategy 2: Numerical Optimization of the Hex Packing
    # We use a penalty function to maximize sum(r) while avoiding overlaps
    
    def objective(params):
        # Unpack: x0, y0, r0, x1, y1, r1, ...
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Penalty for constraints
        penalty = 0.0
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Keep radii positive
            if r < 0:
                return -np.inf # Or huge penalty
            
            # Boundary
            margin = 1e-4
            if x - r < -margin or x + r > 1 + margin or y - r < -margin or y + r > 1 + margin:
                penalty += 10000 * (max(0, r - x) + max(0, r - (1 - x)) + max(0, r - y) + max(0, r - (1 - y)))
        
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    penalty += 10000 * (sum_r - dist)
        
        # We want to maximize sum(radii), so minimize -sum(radii)
        return -np.sum(radii) + penalty

    # Run multiple trials from different initializations
    for attempt in range(5):
        # Initialization: Hex packing perturbed
        centers_init, radii_init = hex_packing()
        
        # Add some noise
        centers_init += np.random.uniform(-0.01, 0.01, size=centers_init.shape)
        radii_init += np.random.uniform(-0.005, 0.005, size=n)
        radii_init = np.abs(radii_init)
        
        # Clip to square
        for i in range(n):
            centers_init[i, 0] = np.clip(centers_init[i, 0], radii_init[i], 1 - radii_init[i])
            centers_init[i, 1] = np.clip(centers_init[i, 1], radii_init[i], 1 - radii_init[i])

        # Initial parameters vector
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        # Optimization
        # Using L-BFGS-B for box constraints might be better, but we handle constraints in penalty
        # Bounds for centers: [0,1], radii: [0, 0.5]
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 10000})
            
            curr_centers = res.x[:2*n].reshape(n, 2)
            curr_radii = res.x[2*n:]
            
            # Ensure radii are non-negative
            curr_radii = np.abs(curr_radii)
            
            # Validate and update best
            # We need to be careful: the penalty function allows small violations
            # We check the actual sum
            curr_sum = np.sum(curr_radii)
            
            # Post-process to strictly satisfy constraints if possible (project radii)
            # If there are overlaps, we might need to reduce radii slightly.
            # However, for the return value, we just want the best valid packing.
            # Let's check validity.
            
            # If not valid, we can try to scale down radii slightly to make it valid
            if not validate_packing(curr_centers, curr_radii):
                # Simple scaling to fix
                max_overlap = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = get_distance(curr_centers[i], curr_centers[j])
                        needed = curr_radii[i] + curr_radii[j]
                        if dist < needed:
                            max_overlap = max(max_overlap, needed - dist)
                
                # Also check boundaries
                for i in range(n):
                    x, y = curr_centers[i]
                    r = curr_radii[i]
                    overlap_b = max(0, r - x) + max(0, r - (1-x)) + max(0, r - y) + max(0, r - (1-y))
                    max_overlap = max(max_overlap, overlap_b)
                
                if max_overlap > 0:
                    # Scale down radii to remove overlap? 
                    # This is tricky because centers might need to move.
                    # But for now, let's just accept the optimizer's result if it's close
                    # or try a different restart.
                    pass

            if curr_sum > best_sum:
                # Verify strictly
                # The optimizer might have found a point with small penalty but still invalid?
                # Let's enforce strict validity by checking overlaps and shrinking radii if needed
                # A simple way to guarantee validity from an approximate solution:
                # Check max overlap and reduce radii uniformly?
                # But we want to maximize sum.
                
                # Let's do a quick check
                valid = True
                for i in range(n):
                    x, y = curr_centers[i]
                    r = curr_radii[i]
                    if x < r or x > 1-r or y < r or y > 1-r:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = get_distance(curr_centers[i], curr_centers[j])
                            if d < curr_radii[i] + curr_radii[j] - 1e-12:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    best_sum = curr_sum
                    best_centers = curr_centers.copy()
                    best_radii = curr_radii.copy()

        except Exception as e:
            pass

    # Fallback: If optimization failed to produce valid packing or low sum,
    # use a robust grid packing with equal radii.
    if best_sum < 2.5:
        # Simple 5x5 grid + 1 circle strategy
        # 5 rows of 5 circles (25 circles) radius 0.1
        # 26th circle? Maybe place in center if gap allows?
        # With r=0.1, grid centers: 0.1, 0.3, 0.5, 0.7, 0.9
        # Center (0.5, 0.5) is occupied.
        # Maybe place at (0.5, 0.5) with smaller radius?
        # Distance to neighbors (0.3, 0.5) is 0.2. Sum radii 0.1+r_new <= 0.2 => r_new <= 0.1.
        # So we can't fit another 0.1 radius circle.
        
        # Let's try a perturbation of the grid to squeeze one in?
        # Or use the best result from optimizer even if slightly less sum but valid.
        
        # Let's generate a valid equal-radius packing for 26 circles with r ~ 0.095
        r = 0.095
        centers_eq = []
        # 6 rows of hexagonal packing
        # Row 1: 5 circles
        # Row 2: 5 circles (shifted)
        # ...
        # Total 26
        
        y = r
        count = 0
        row_parity = 0
        while count < 26:
            shift = r * row_parity # 0 or r
            x = r + shift
            while x <= 1 - r and count < 26:
                centers_eq.append([x, y])
                count += 1
                x += 2 * r
            y += r * math.sqrt(3)
            row_parity = 1 - row_parity
        
        if len(centers_eq) >= 26:
            best_centers = np.array(centers_eq[:26])
            best_radii = np.array([r] * 26)
            best_sum = np.sum(best_radii)

    # Final validation and adjustment
    # Ensure the returned packing is strictly valid
    if not validate_packing(best_centers, best_radii):
        # Try to fix by reducing radii slightly
        factor = 1.0
        while not validate_packing(best_centers, best_radii * factor) and factor > 0.8:
            factor -= 0.01
        best_radii = best_radii * factor
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum