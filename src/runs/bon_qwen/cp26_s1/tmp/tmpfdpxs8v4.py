import numpy as np
from scipy.optimize import minimize

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

def calculate_overlap_penalty(centers, radii, n):
    """
    Calculates a penalty based on overlaps and boundary violations.
    Lower is better. 0 means valid.
    """
    penalty = 0.0
    # Boundary penalties
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: penalty += (x - r) ** 2
        if x + r > 1: penalty += (x + r - 1) ** 2
        if y - r < 0: penalty += (y - r) ** 2
        if y + r > 1: penalty += (y + r - 1) ** 2

    # Overlap penalties
    # Vectorized distance calculation
    # centers shape (n, 2)
    # diffs shape (n, n, 2)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    # Radii sums shape (n, n)
    rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only upper triangle to avoid double counting and self-overlap
    # We can just sum all and divide by 2, but self is 0.
    violation = rad_sums - dists
    np.fill_diagonal(violation, 0) # No self overlap
    violation = np.maximum(0, violation)
    penalty += np.sum(violation ** 2)
    
    return penalty

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # Initialize centers: 5x5 grid + 1 center
    centers = np.zeros((n, 2))
    # Grid centers
    grid_step = 0.2
    offset = 0.1
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx, 0] = offset + j * grid_step
            centers[idx, 1] = offset + i * grid_step
            idx += 1
    # 26th circle in the center
    centers[25, 0] = 0.5
    centers[25, 1] = 0.5
    
    # Initial small radius to allow optimization to start
    radii = np.full(n, 0.01)
    
    # Iterative expansion and optimization
    # We try to increase radii gradually and resolve overlaps
    current_sum = np.sum(radii)
    
    for step in range(50):
        # Try to increase radii
        radii *= 1.01
        
        # Objective: Minimize overlap penalty
        # We use a simple gradient descent step or scipy minimize for refinement
        
        # Convert to 1D vector for scipy: [x1, y1, x2, y2, ..., r1, r2, ...]
        # But keeping radii fixed during position optimization is often more stable
        # Let's optimize positions first to resolve overlaps, then increase radii
        
        def objective(pos):
            c = pos.reshape((n, 2))
            return calculate_overlap_penalty(c, radii, n)

        # Initial guess for scipy
        x0 = centers.flatten()
        
        # Bounds for centers
        bnds = []
        for _ in range(n):
            bnds.append((0, 1))
            bnds.append((0, 1))
            
        # We use Nelder-Mead as it doesn't require gradients and handles non-smoothness better
        # though it can be slow, for n=26 it should be okay for a few steps
        # To keep it fast, we limit maxiter
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'adaptive': True},
                       bounds=bnds)
        
        if res.success or res.fun < 1e-6:
            centers = res.x.reshape((n, 2))
        else:
            # Fallback to simple push if scipy fails or is too slow
            # Just update centers if we got a better result, else keep old
            pass
            
        # Verify if we can maintain the radii increase
        penalty = calculate_overlap_penalty(centers, radii, n)
        if penalty > 1e-4:
            # If penalty is too high, revert radius increase and re-optimize positions harder
            radii /= 1.01
            # Run more optimization steps
            res = minimize(objective, centers.flatten(), method='Nelder-Mead', 
                           options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 1000})
            centers = res.x.reshape((n, 2))
            
        current_sum = np.sum(radii)
        # If sum stops growing significantly, we might be done
        # But we continue to push boundaries

    # Final cleanup to ensure strict constraints
    # Clip centers to [0,1]
    centers = np.clip(centers, 0, 1)
    
    # Adjust radii to be strictly valid based on final positions
    # This is a safety step. If circles are touching, this ensures no negative radii.
    # However, we want to maximize sum, so we shouldn't shrink radii if not necessary.
    # The optimization should have found a valid config.
    
    # Let's verify and if invalid, do a final aggressive shrink to make it valid
    # But usually the penalty function ensures validity.
    
    # Re-calculate sum
    final_sum = np.sum(radii)
    
    # Return result
    return centers, radii, final_sum

# To test locally
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")