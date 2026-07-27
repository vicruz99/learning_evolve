# sol_000101 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bf51a1cd) state=76a41450 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a simulated annealing approach with a hexagonal lattice initialization.
    """
    np.random.seed(42)
    N = 26
    # Initialize with a hexagonal lattice scaled to fit in [0,1]x[0,1]
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing arrangement
    r_init = 0.09
    row = 0
    col = 0
    count = 0
    
    # Estimate rows and cols for hex grid
    # Approximate spacing
    # x_spacing = 2*r, y_spacing = sqrt(3)*r
    # Shift odd rows by r
    y_step = np.sqrt(3) * r_init
    
    while count < N:
        x = col * 2 * r_init + r_init
        y = row * y_step + r_init
        
        if row % 2 == 1:
            x += r_init
            
        if x + r_init <= 1.0 and y + r_init <= 1.0:
            centers[count] = [x, y]
            radii[count] = r_init
            count += 1
            col += 1
            if col * 2 * r_init + r_init > 1.0:
                col = 0
                row += 1
        else:
            col = 0
            row += 1

    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)

    # Optimization parameters
    temp = 0.01
    min_temp = 1e-9
    cooling_rate = 0.9995
    iterations = 20000
    
    # Helper to check validity and compute penalty
    def check_validity(centers, radii):
        # Boundary checks
        for i in range(N):
            if radii[i] < 0 or centers[i][0] < radii[i] or centers[i][0] + radii[i] > 1.0 + 1e-9:
                return False
            if centers[i][1] < radii[i] or centers[i][1] + radii[i] > 1.0 + 1e-9:
                return False
        
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    current_centers = centers.copy()
    current_radii = radii.copy()
    current_sum = np.sum(current_radii)

    for iteration in range(iterations):
        # Choose a random circle to perturb
        idx = np.random.randint(N)
        
        # Decide type of perturbation: position or radius
        if np.random.random() < 0.7:
            # Perturb position
            delta = np.random.uniform(-temp, temp, 2)
            new_center = current_centers[idx] + delta
            new_centers = current_centers.copy()
            new_centers[idx] = new_center
            new_radii = current_radii.copy()
        else:
            # Perturb radius
            delta_r = np.random.uniform(-temp, temp)
            new_radii = current_radii.copy()
            new_radii[idx] = max(0, current_radii[idx] + delta_r)
            new_centers = current_centers.copy()

        # Check validity
        if check_validity(new_centers, new_radii):
            new_sum = np.sum(new_radii)
            delta_sum = new_sum - current_sum
            
            # Accept if better or with probability exp(delta/T)
            if delta_sum > 0 or np.random.random() < np.exp(delta_sum / max(temp, 1e-12)):
                current_centers = new_centers
                current_radii = new_radii
                current_sum = new_sum
                
                # Update best
                if current_sum > best_sum:
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
                    best_sum = current_sum
        
        # Cool down
        temp *= cooling_rate
        if temp < min_temp:
            break
            
    # Return the best found configuration
    # Ensure final validation
    if check_validity(best_centers, best_radii):
        return best_centers, best_radii, np.sum(best_radii)
    else:
        # Fallback to initialization if optimization failed (should not happen)
        return centers, radii, np.sum(radii)
