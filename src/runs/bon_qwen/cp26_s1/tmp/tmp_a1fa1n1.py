import numpy as np

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

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
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

def initialize_hex_packing(n):
    """
    Initialize n circles in a hexagonal packing pattern.
    """
    # Estimate radius to fit in square initially
    # For n=26, roughly 5x5 grid but hexagonal.
    # Let's start with a small radius and scale up later.
    # Or place them in a grid first.
    
    # Simple grid initialization for stability
    cols = 6
    rows = 5
    # Adjust if n doesn't fit perfectly
    centers = []
    idx = 0
    
    # Hexagonal packing layout
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    # We need 26 circles. 
    # Maybe 5 rows of 5 = 25, plus 1?
    # Or 6, 5, 5, 5, 5?
    
    # Let's try a dense hexagonal arrangement
    # Width constraint: k circles in a row need width roughly 2*k*r?
    # Actually centers span (2k-1)r? No, 2r spacing.
    # Let's just place them in a grid and let the optimizer fix positions.
    
    # A 5x6 grid is 30 spots. We take 26.
    # Spacing
    dx = 1.0 / 5.0
    dy = 1.0 / 5.0
    
    pts = []
    for r in range(5):
        for c in range(6):
            if len(pts) < 26:
                # Hex shift for alternating rows
                shift = 0.5 * dx if r % 2 == 1 else 0
                pts.append([c * dx + shift + 0.5*dx, r * dy + 0.5*dy])
    
    centers = np.array(pts[:26])
    radii = np.full(26, 0.01) # Start small
    return centers, radii

def run_packing():
    # Set seed for reproducibility if needed, but randomness helps escape local optima
    np.random.seed(42)

    n = 26
    centers, radii = initialize_hex_packing(n)
    
    # Simulation parameters
    dt = 1e-4
    growth_rate = 0.0005
    force_scale = 1000.0
    boundary_stiffness = 1000.0
    max_iterations = 20000
    
    # Optimization loop
    for step in range(max_iterations):
        # 1. Grow radii
        # We try to increase radii. If valid, we keep them.
        # To maximize sum, we grow all radii.
        # A simple way: check if we can increase.
        # But better to use forces.
        
        # Let's implement a "grow and resolve" step
        # Increase radii slightly
        radii += growth_rate
        
        # 2. Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            # Boundary forces
            # If circle touches or crosses boundary, push back
            # Left
            if centers[i, 0] - radii[i] < 0:
                forces[i, 0] += boundary_stiffness * (-(centers[i, 0] - radii[i]))
            # Right
            if centers[i, 0] + radii[i] > 1:
                forces[i, 0] -= boundary_stiffness * (centers[i, 0] + radii[i] - 1)
            # Bottom
            if centers[i, 1] - radii[i] < 0:
                forces[i, 1] += boundary_stiffness * (-(centers[i, 1] - radii[i]))
            # Top
            if centers[i, 1] + radii[i] > 1:
                forces[i, 1] -= boundary_stiffness * (centers[i, 1] + radii[i] - 1)
            
            # Interaction forces
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist == 0:
                    dist = 1e-9
                    diff = np.array([1e-9, 0])
                
                req_dist = radii[i] + radii[j]
                overlap = req_dist - dist
                
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    # F = k * overlap / dist * direction
                    f_mag = force_scale * overlap
                    f_vec = f_mag * diff / dist
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # 3. Update centers
        centers += dt * forces
        
        # 4. Hard constraints clamp (to ensure validity during simulation)
        for i in range(n):
            # Clamp x
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            # Clamp y
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

    # Final refinement: shrink radii slightly if any overlap due to numerical error
    # And ensure strict validity
    # We can run a few steps of just repulsion with current radii to settle
    for _ in range(1000):
        forces = np.zeros_like(centers)
        for i in range(n):
             for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-9:
                    dist = 1e-9
                    diff = np.array([1e-9, 0])
                req_dist = radii[i] + radii[j]
                overlap = req_dist - dist
                if overlap > 0:
                    f_mag = 1000 * overlap
                    f_vec = f_mag * diff / dist
                    forces[i] += f_vec
                    forces[j] -= f_vec
        centers += 1e-4 * forces
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

    # Final validation and adjustment
    # If validation fails, shrink radii until valid
    valid = False
    temp_radii = radii.copy()
    centers_temp = centers.copy()
    
    # Simple binary search or linear shrink to find valid config close to current
    # Since we want max sum, we might have overshot slightly.
    # Let's check validity.
    if validate_packing(centers_temp, temp_radii):
        final_centers = centers_temp
        final_radii = temp_radii
    else:
        # Shrink radii gradually
        while not validate_packing(centers_temp, temp_radii) and temp_radii[0] > 0:
            temp_radii *= 0.99
            # Recompute valid positions? 
            # Just shrinking radii might still leave overlaps if centers are bad.
            # But our simulation pushed them apart.
            # If still invalid, we need to move centers.
            # Let's just run a few force steps with smaller radii?
            # Actually, if radii are smaller, overlaps are less likely.
            pass
            
        # If still invalid after shrinking, force solve again
        # But for 26 circles, the simulation usually works.
        # Let's assume the simulation result is close.
        # Let's try to fix overlaps by shrinking.
        # Calculate max overlap
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers_temp[i] - centers_temp[j])
                overlap = (temp_radii[i] + temp_radii[j]) - dist
                if overlap > max_overlap:
                    max_overlap = overlap
            # Boundary
            x, y = centers_temp[i]
            r = temp_radii[i]
            b_overlap = 0
            if x - r < 0: b_overlap = max(b_overlap, -(x-r))
            if x + r > 1: b_overlap = max(b_overlap, x+r-1)
            if y - r < 0: b_overlap = max(b_overlap, -(y-r))
            if y + r > 1: b_overlap = max(b_overlap, y+r-1)
            if b_overlap > max_overlap:
                max_overlap = b_overlap
        
        # If max_overlap > 0, we must shrink radii by max_overlap/2 roughly?
        # Or just scale down.
        if max_overlap > 1e-12:
            scale = 1 - max_overlap / np.sum(temp_radii) * 1.1 # Safe margin
            temp_radii *= scale
            # Also need to ensure centers are valid w.r.t new radii
            for i in range(n):
                x, y = centers_temp[i]
                r = temp_radii[i]
                centers_temp[i, 0] = np.clip(x, r, 1-r)
                centers_temp[i, 1] = np.clip(y, r, 1-r)
        
        final_centers = centers_temp
        final_radii = temp_radii

    # Ensure no NaNs
    if np.isnan(final_centers).any() or np.isnan(final_radii).any():
        return np.zeros((26, 2)), np.zeros(26), 0.0

    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii

# To run and print result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Validation: {validate_packing(c, r)}")