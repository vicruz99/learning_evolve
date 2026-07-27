import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n_circles = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Helper to create a hexagonal initial configuration
    def create_hexagonal_config(n, perturbation=0.0):
        # Try to arrange in rows. 
        # Approximate row height for hex packing: sqrt(3)/2 * 2r = sqrt(3)r. 
        # But we don't know r yet. Let's assume r=0.1 roughly.
        # We can just place them in a grid and shift rows.
        
        # Let's try to fit n circles in a roughly hexagonal pattern
        # Estimate r ~ 0.1. 
        # Width ~ 1, so ~5 circles per row.
        # Rows needed ~ 26/5 = 5.2 -> 6 rows.
        
        # Let's try a pattern like 5, 5, 5, 5, 5, 1? Or 6, 5, 5, 5, 5?
        # Let's just generate a hex grid and take first n.
        
        cols = 6
        rows = 5
        # Spacing
        dx = 1.0 / (cols + 1) # slightly spaced out
        dy = 1.0 / (rows + 1)
        
        centers = []
        count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if count >= n:
                    break
                x = (c_idx + 1) * dx
                y = (r_idx + 1) * dy
                # Shift odd rows
                if r_idx % 2 == 1:
                    x += dx / 2.0
                
                # Add perturbation
                if perturbation > 0:
                    x += np.random.uniform(-perturbation, perturbation)
                    y += np.random.uniform(-perturbation, perturbation)
                
                # Clamp to [0.1, 0.9] to be safe initially
                x = np.clip(x, 0.1, 0.9)
                y = np.clip(y, 0.1, 0.9)
                
                centers.append([x, y])
                count += 1
            if count >= n:
                break
        
        # If we didn't get enough, fill with random
        while len(centers) < n:
            centers.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)])
        
        # Convert to array
        centers_arr = np.array(centers[:n])
        radii_arr = np.full(n, 0.05) # Start small
        return centers_arr, radii_arr

    def objective(vars):
        # vars is [x1, y1, r1, x2, y2, r2, ...]
        radii = vars[2::3]
        return -np.sum(radii) # Minimize negative sum

    def constraint_bounds(vars):
        # Returns an array of constraint values. 
        # We require value >= 0.
        # But SLSQP handles bounds separately.
        # We just need non-overlap constraints here?
        # Actually, SLSQP bounds are box bounds.
        # Non-overlap is nonlinear.
        
        # We will handle boundary constraints via bounds or explicit constraints.
        # Explicit constraints are safer for non-linear interactions.
        # But for speed, let's use bounds for box and explicit for overlap.
        
        # However, passing 300+ constraints to SLSQP might be slow.
        # Let's stick to explicit constraints for overlap.
        
        # Extract radii and centers
        radii = vars[2::3]
        centers = np.column_stack((vars[0::3], vars[1::3]))
        
        cons = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                # dist^2 >= (r_i + r_j)^2
                # dist^2 - (r_i + r_j)^2 >= 0
                dist_sq = np.sum((centers[i] - centers[j])**2)
                r_sum = radii[i] + radii[j]
                cons.append(dist_sq - r_sum**2)
        
        # Also boundary constraints explicitly to be safe, 
        # though bounds usually handle x,r,y.
        # But r is coupled with x.
        # Bounds: 0 <= x <= 1, 0 <= y <= 1, 0 <= r <= 0.5
        # We need x - r >= 0 => x >= r. This is not a box bound.
        # So we must add boundary constraints.
        
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            cons.append(x - r)
            cons.append(y - r)
            cons.append(1.0 - x - r)
            cons.append(1.0 - y - r)
            
        return np.array(cons)

    # We can define constraints as dictionaries for SLSQP
    # But passing a function that returns array is also supported if type='ineq'
    # Actually, for many constraints, dict list might be better?
    # Let's use a single dict with 'fun' returning array.
    
    non_overlap_constraint = {
        'type': 'ineq',
        'fun': constraint_bounds
    }

    # Bounds for x, y, r
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius is 0.5)
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Run multiple trials
    best_score = -np.inf
    
    # Trial 1: Hexagonal init
    centers, radii = create_hexagonal_config(n_circles, perturbation=0.05)
    x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    
    # Run optimizer
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=non_overlap_constraint, 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success or res.fun < best_score: # min fun is neg sum
            # Check validity manually just in case
            c = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            curr_sum = np.sum(r)
            if curr_sum > best_score:
                # Validate strictly
                if validate_packing(np.column_stack((c, y)), r):
                    best_score = curr_sum
                    best_centers = np.column_stack((c, y))
                    best_radii = r.copy()
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Trial 2: Random init
    for _ in range(3):
        centers = np.random.rand(n_circles, 2) * 0.6 + 0.2
        radii = np.full(n_circles, 0.05)
        x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=non_overlap_constraint, 
                           options={'maxiter': 500, 'ftol': 1e-12})
            c = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            curr_sum = np.sum(r)
            # Since objective is -sum, lower res.fun is better. 
            # But we check sum directly.
            if curr_sum > best_score:
                if validate_packing(np.column_stack((c, y)), r):
                    best_score = curr_sum
                    best_centers = np.column_stack((c, y))
                    best_radii = r.copy()
        except:
            pass

    # If no valid packing found (unlikely), fallback to grid
    if best_centers is None:
        # Fallback 5x5 grid + 1 small
        centers = []
        radii = []
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
                radii.append(0.1)
        # Add 26th circle in a gap? 
        # Just add a tiny one at 0.5, 0.5
        centers.append([0.5, 0.5])
        radii.append(0.001) # Tiny to be valid
        best_centers = np.array(centers)
        best_radii = np.array(radii)
        best_score = np.sum(best_radii)

    # Final refinement: Try to expand radii locally
    # If the optimizer stopped early, we might be able to grow them.
    # But SLSQP should have found a local optimum.
    
    # One last check: ensure strict validity with epsilon
    # The validate function uses 1e-12 tolerance.
    # We might need to shrink slightly if on the edge.
    
    # Let's verify the best solution
    if not validate_packing(best_centers, best_radii):
        print("Warning: Final packing invalid, attempting fix...")
        # Simple fix: shrink radii slightly
        factor = 0.99
        while not validate_packing(best_centers, best_radii * factor) and factor > 0.5:
            factor -= 0.01
        best_radii = best_radii * factor
        best_score = np.sum(best_radii)

    return best_centers, best_radii, best_score

# Include validation function as provided in prompt to ensure access
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True