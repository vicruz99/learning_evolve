import numpy as np
from scipy.optimize import minimize
import random

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

def generate_initial_packing(n_circles, seed=0):
    """
    Generate an initial packing using a hexagonal-like grid pattern.
    """
    rng = np.random.RandomState(seed)
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Hexagonal grid parameters
    # We try to fit points in a hexagonal pattern
    # Approximate density packing
    # 5x5 grid is 25. We need 26.
    # Let's try to place them in a slightly irregular grid to allow expansion
    
    # Start with a dense random packing or grid
    # A simple grid 5x5 + 1 in middle is good but symmetric.
    # Let's create a perturbed grid.
    
    # 5 rows, roughly 5-6 columns
    rows = 6
    cols = 5
    # Spacing
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_circles:
                break
            x = (c + 1) * dx + rng.uniform(-0.1, 0.1) * dx
            y = (r + 1) * dy + rng.uniform(-0.1, 0.1) * dy
            # Clamp to center to be safe initially
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            centers[idx] = [x, y]
            radii[idx] = 0.01 # Small initial radius
            idx += 1
            
    if idx < n_circles:
        # Fill remaining if needed (shouldn't happen with 6x5=30)
        for k in range(idx, n_circles):
            centers[k] = rng.uniform(0.1, 0.9, 2)
            radii[k] = 0.01
            
    return centers[:n_circles], radii[:n_circles]

def expand_and_relax(centers, radii, expansion_factor=1.01, max_iters=500, step_size=0.05):
    """
    Iteratively expand radii and resolve collisions using force-directed relaxation.
    """
    n = len(radii)
    centers = centers.copy()
    radii = radii.copy()
    
    # Precompute indices for neighbor interactions
    # For small N, O(N^2) is fine
    
    for _ in range(max_iters):
        # 1. Expand radii
        radii *= expansion_factor
        
        # 2. Relaxation steps to resolve overlaps
        relaxed = False
        for _ in range(100): # Inner relaxation steps
            forces = np.zeros_like(centers)
            overlap_count = 0
            
            # Check circle-circle overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    diff = centers[i] - centers[j]
                    dist = np.sqrt(np.sum(diff**2))
                    r_sum = radii[i] + radii[j]
                    
                    if dist < r_sum and dist > 1e-9:
                        overlap = r_sum - dist
                        # Normalize direction
                        dir_vec = diff / dist
                        # Apply repulsive force proportional to overlap
                        # Split force equally
                        force_mag = overlap * 0.5 
                        forces[i] += dir_vec * force_mag
                        forces[j] -= dir_vec * force_mag
                        overlap_count += 1
            
            # Check boundary overlaps
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                
                # Left wall
                if x - r < 0:
                    forces[i, 0] += (r - x)
                # Right wall
                if x + r > 1:
                    forces[i, 0] -= (x + r - 1)
                # Bottom wall
                if y - r < 0:
                    forces[i, 1] += (r - y)
                # Top wall
                if y + r > 1:
                    forces[i, 1] -= (y + r - 1)
                overlap_count += (1 if (x-r < 0) else 0) + (1 if (x+r > 1) else 0) + \
                                (1 if (y-r < 0) else 0) + (1 if (y+r > 1) else 0)
            
            if overlap_count == 0:
                break
                
            # Apply forces
            centers += forces * step_size
            
            # Clamp centers to [0, 1] just in case numerical errors push them out
            centers = np.clip(centers, 0, 1)
        
        if overlap_count == 0:
            # If valid, continue expanding
            continue
        else:
            # If stuck, shrink slightly and try again or break
            # For this heuristic, we just accept the current state if we can't resolve
            # But to ensure validity, we might need to shrink.
            # However, the loop continues to try.
            pass

    # Final check and cleanup
    # If overlaps still exist, shrink radii to satisfy constraints
    # This is a safety measure
    radii_current = radii.copy()
    centers_current = centers.copy()
    
    # Iteratively shrink to resolve any remaining overlaps
    for _ in range(100):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers_current[i] - centers_current[j]
                dist = np.sqrt(np.sum(diff**2))
                r_sum = radii_current[i] + radii_current[j]
                if dist < r_sum:
                    valid = False
                    # Shrink both slightly
                    shrink = (r_sum - dist) / 2.0 + 1e-6
                    radii_current[i] -= shrink
                    radii_current[j] -= shrink
            # Boundary check
            x, y = centers_current[i]
            r = radii_current[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                r_new = min(x, 1-x, y, 1-y)
                if r_new < r:
                    radii_current[i] = r_new
        
        if valid:
            break
            
    return centers_current, radii_current

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    n_circles = 26
    
    # Try multiple random seeds to find a good local optimum
    seeds = [0, 1, 10, 42, 123, 456, 789, 1000, 2000, 3000]
    
    for seed in seeds:
        # 1. Initialize
        centers, radii = generate_initial_packing(n_circles, seed=seed)
        
        # 2. Expand and Relax
        # Use a coarse expansion then finer steps
        # We can chain multiple expansions
        temp_centers, temp_radii = centers.copy(), radii.copy()
        
        # Phase 1: Fast growth
        for _ in range(100):
            temp_radii *= 1.02
            # Quick relaxation
            for _ in range(50):
                forces = np.zeros_like(temp_centers)
                has_overlap = False
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        diff = temp_centers[i] - temp_centers[j]
                        dist = np.sqrt(np.sum(diff**2))
                        r_sum = temp_radii[i] + temp_radii[j]
                        if dist < r_sum and dist > 1e-9:
                            overlap = r_sum - dist
                            dir_vec = diff / dist
                            forces[i] += dir_vec * overlap * 0.5
                            forces[j] -= dir_vec * overlap * 0.5
                            has_overlap = True
                    
                    # Boundaries
                    x, y = temp_centers[i]
                    r = temp_radii[i]
                    if x - r < 0: forces[i, 0] += (r - x)
                    if x + r > 1: forces[i, 0] -= (x + r - 1)
                    if y - r < 0: forces[i, 1] += (r - y)
                    if y + r > 1: forces[i, 1] -= (y + r - 1)
                
                if not has_overlap:
                    break
                temp_centers += forces * 0.05
                temp_centers = np.clip(temp_centers, 0, 1)
        
        # Phase 2: Fine tuning
        # Use the expand_and_relax function or similar logic
        # Let's just run the robust function defined above
        temp_centers, temp_radii = expand_and_relax(temp_centers, temp_radii, expansion_factor=1.001, max_iters=500, step_size=0.02)
        
        # Validate and calculate sum
        # The expand_and_relax ensures validity (mostly)
        # But let's double check and fix radii if needed
        # We can project radii to valid max radius for current centers
        
        # Recalculate max possible radii for current centers
        # This is a linear problem but simple approximation works:
        # r_i = min(dist to boundary, min(dist to j)/2)
        # However, this is only valid if we ignore the coupling.
        # But for a final valid packing, we can just use the radii from the process
        # and ensure they are valid.
        
        # Safety clamp
        for i in range(n_circles):
            x, y = temp_centers[i]
            r = temp_radii[i]
            max_r_boundary = min(x, 1-x, y, 1-y)
            temp_radii[i] = min(temp_radii[i], max_r_boundary)
            
            for j in range(i + 1, n_circles):
                diff = temp_centers[i] - temp_centers[j]
                dist = np.sqrt(np.sum(diff**2))
                max_r_pair = dist / 2.0
                # This is tricky because reducing r_i affects r_j.
                # But if we just take min, we might reduce too much.
                # However, the previous step should have resolved this.
                pass
        
        # Final validation fix:
        # If any overlap exists, shrink radii
        for _ in range(10):
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    diff = temp_centers[i] - temp_centers[j]
                    dist = np.sqrt(np.sum(diff**2))
                    r_sum = temp_radii[i] + temp_radii[j]
                    if dist < r_sum:
                        shrink = (r_sum - dist) / 2.0 + 1e-9
                        temp_radii[i] -= shrink
                        temp_radii[j] -= shrink
                
                x, y = temp_centers[i]
                r = temp_radii[i]
                if x - r < 0: temp_radii[i] = x
                if x + r > 1: temp_radii[i] = 1 - x
                if y - r < 0: temp_radii[i] = y
                if y + r > 1: temp_radii[i] = 1 - y
                temp_radii[i] = max(0, temp_radii[i])

        current_sum = np.sum(temp_radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = temp_centers.copy()
            best_radii = temp_radii.copy()
            
    # Final validation
    if not validate_packing(best_centers, best_radii):
        print("Warning: Final packing invalid, attempting repair...")
        # Repair by shrinking
        for i in range(n_circles):
            x, y = best_centers[i]
            r = best_radii[i]
            best_radii[i] = min(r, x, 1-x, y, 1-y)
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < best_radii[i] + best_radii[j]:
                    # Shrink both
                    shrink = (best_radii[i] + best_radii[j] - dist) / 2.0
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    best_radii[i] = max(0, best_radii[i])
                    best_radii[j] = max(0, best_radii[j])
    
    best_sum = np.sum(best_radii)
    return best_centers, best_radii, float(best_sum)

# Example usage / validation
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(validate_packing(centers, radii))