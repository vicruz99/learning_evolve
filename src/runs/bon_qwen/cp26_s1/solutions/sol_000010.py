# sol_000010 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9f77b693) state=7af97ab1 sum of radii=2.136401 correctness=1.0
# stdout(first 200): Circles 1 and 2 overlap: dist=0.1859742959990647, r1+r2=0.18609752572376642 Circles 21 and 25 overlap: dist=0.09662200987698985, r1+r2=0.09730843627674712
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

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

def get_initial_config():
    """
    Generates an initial hexagonal packing configuration for 26 circles.
    Uses 6 rows to distribute circles more evenly and avoid strict row-width constraints.
    """
    centers = []
    # Configuration: 6 rows with circle counts 5, 5, 5, 5, 5, 1 is bad.
    # Better: 5, 5, 6, 5, 5, 0? No.
    # Let's try a dense hexagonal pattern that fits in the square.
    # We will place centers and then optimize.
    
    # A robust initial grid: 6 columns x 5 rows is 30 circles. We pick 26.
    # Or better, a specific hexagonal layout.
    # Let's try to fit 26 circles with radius ~0.09 initially.
    
    # Hexagonal lattice parameters
    # Rows spaced by sqrt(3)*r, shifted by r/2 or r
    # Let's assume r ~ 0.09.
    
    # Let's create a list of centers
    # 5 rows of 5 circles = 25. Plus 1 circle?
    # 5x5 grid centers:
    # x in [0.1, 0.3, 0.5, 0.7, 0.9]
    # y in [0.1, 0.3, 0.5, 0.7, 0.9]
    # This leaves gaps.
    
    # Let's use a 6x5 grid subset or perturbed hex lattice.
    # 6 rows, alternating 5 and 4 circles? 6*4.5 = 27.
    # Rows: 5, 4, 5, 4, 5, 3? Sum = 26.
    
    # Let's try a simple initialization that is dense.
    # 6 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 4 circles
    # Row 5: 2 circles?
    # Total 26.
    
    # Actually, just a random dense placement optimized later is often fine,
    # but structure helps.
    
    # Let's place them in a hexagonal pattern.
    # Spacing d.
    # x coords: d/2, 3d/2, ...
    # y coords: d*sqrt(3)/2, 3d*sqrt(3)/2, ...
    
    # Let's estimate d. 26 circles area approx 0.8. d approx 0.2.
    d = 0.2
    h = d * np.sqrt(3) / 2
    
    centers = []
    row_counts = [5, 5, 5, 5, 4, 2] # Sum 26? 5+5+5+5+4+2 = 26.
    # Let's adjust to be more balanced: 5, 4, 5, 4, 5, 3 -> 26.
    row_counts = [5, 4, 5, 4, 5, 3]
    
    # Vertical spacing needs to fit in 1.0
    # 6 rows. Height approx 5*h + 2r.
    # Let's just place them and let optimizer fix it.
    
    for r_idx, count in enumerate(row_counts):
        y = 0.1 + r_idx * h # Start near 0.1
        # Horizontal span
        # If count=5, span 4*d. Width 4d + 2r.
        # Let's center them.
        total_width = (count - 1) * d
        start_x = (1.0 - total_width) / 2
        
        # Shift odd rows
        shift = 0
        if r_idx % 2 == 1:
            shift = d / 2
            
        for c in range(count):
            x = start_x + c * d + shift
            # Clamp to bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
            
    centers = np.array(centers)
    # Ensure we have exactly 26
    # If row_counts sum is not 26, adjust.
    # 5+4+5+4+5+3 = 26. Correct.
    
    return centers

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    centers = get_initial_config()
    # Initial radii small enough to not overlap significantly
    radii = np.ones(n) * 0.05
    
    # Flatten variables for optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-4, 0.5)) # r
    
    # Objective: Maximize sum of radii => Minimize negative sum of radii
    # We use a penalty method for constraints.
    # Penalty for overlap and boundary violation.
    
    def objective(vars):
        centers_opt = vars[0:2*n].reshape(n, 2)
        radii_opt = vars[2*n:]
        
        # Negative sum of radii
        loss = -np.sum(radii_opt)
        
        # Penalty term
        penalty = 0.0
        alpha = 1000.0 # Penalty weight
        
        # Boundary penalties
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            # Distance to boundaries
            d_left = x - r
            d_right = 1.0 - x - r
            d_bottom = y - r
            d_top = 1.0 - y - r
            
            if d_left < 0: penalty += alpha * (d_left**2)
            if d_right < 0: penalty += alpha * (d_right**2)
            if d_bottom < 0: penalty += alpha * (d_bottom**2)
            if d_top < 0: penalty += alpha * (d_top**2)
            
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += alpha * (overlap**2)
                    
        return loss + penalty

    # Use SLSQP or L-BFGS-B. SLSQP handles bounds well.
    # However, penalty method turns constraints into objective, so bounds are just for variables.
    # Let's use L-BFGS-B as it is often faster for smooth approximations.
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 5000})
    
    final_vars = res.x
    final_centers = final_vars[0:2*n].reshape(n, 2)
    final_radii = final_vars[2*n:]
    
    # Post-processing to ensure strict validity and slight cleanup
    # Clip radii to be non-negative
    final_radii = np.maximum(final_radii, 0.0)
    
    # Ensure centers are within bounds considering radii
    # This might be slightly violated by numerical error in penalty method, 
    # so we project.
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        final_centers[i] = [x, y]
        
    # Final validation and small adjustment if needed
    # If validation fails, we might need to reduce radii slightly.
    if not validate_packing(final_centers, final_radii):
        # Reduce radii slightly until valid
        scale = 1.0
        for _ in range(100):
            scale *= 0.99
            test_radii = final_radii * scale
            if validate_packing(final_centers, test_radii):
                final_radii = test_radii
                break
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
