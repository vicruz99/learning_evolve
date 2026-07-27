import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses equal radii assumption and numerical optimization.
    """
    n_circles = 26
    best_r = 0.0
    best_centers = None
    
    # Binary search range for radius
    r_low = 0.02
    r_high = 0.15
    tolerance = 1e-5
    
    # Optimization parameters
    n_restarts = 10
    optim_method = 'L-BFGS-B'
    
    # Pre-allocate arrays for performance
    # centers shape (26, 2) -> flattened to 52 dims for optimizer
    
    def penalty_function(centers_flat, r):
        centers = centers_flat.reshape(-1, 2)
        penalty = 0.0
        
        # Boundary penalties
        # Constraint: r <= x <= 1-r and r <= y <= 1-r
        # Violation: if x < r, penalty (x-r)^2. If x > 1-r, penalty (x-(1-r))^2.
        # Note: Using max(0, violation) squared ensures non-negativity
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Left wall
        viol = r - x
        mask = viol > 0
        penalty += np.sum(viol[mask]**2)
        
        # Right wall
        viol = x - (1.0 - r)
        mask = viol > 0
        penalty += np.sum(viol[mask]**2)
        
        # Bottom wall
        viol = r - y
        mask = viol > 0
        penalty += np.sum(viol[mask]**2)
        
        # Top wall
        viol = y - (1.0 - r)
        mask = viol > 0
        penalty += np.sum(viol[mask]**2)
        
        # Overlap penalties
        # dist >= 2r. Violation if dist < 2r.
        # penalty = (2r - dist)^2
        # Vectorized distance calculation
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask out diagonal and upper triangle to avoid double counting and self-check
        mask = np.triu(np.ones((n_circles, n_circles)), k=1).astype(bool)
        
        # Overlap amount
        overlap = 2.0 * r - dist
        # Only positive overlaps contribute
        overlap = np.maximum(0, overlap)
        
        # Sum squared overlaps
        penalty += np.sum(overlap[mask]**2)
        
        return penalty

    # Binary search loop
    for _ in range(40): # Sufficient iterations for precision
        if r_high - r_low < tolerance:
            break
            
        r_mid = (r_low + r_high) / 2
        
        # Try to find a valid configuration for r_mid
        min_penalty_found = np.inf
        
        # Multiple random restarts to avoid local minima
        for _ in range(n_restarts):
            # Random initialization within valid bounds if possible, or just [0,1]
            # Using [r, 1-r] helps start valid, but might be empty if r > 0.5
            # Here r is small, so valid region exists.
            x = np.random.uniform(r_mid, 1.0 - r_mid, size=n_circles)
            y = np.random.uniform(r_mid, 1.0 - r_mid, size=n_circles)
            centers_flat = np.concatenate([x, y])
            
            # Bounds for optimizer: [0, 1] for each coordinate
            bounds = [(0.0, 1.0)] * (2 * n_circles)
            
            result = minimize(
                fun=penalty_function,
                x0=centers_flat,
                args=(r_mid,),
                method=optim_method,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            if result.fun < min_penalty_found:
                min_penalty_found = result.fun
                if min_penalty_found < 1e-9: # Valid configuration found
                    best_r = r_mid
                    best_centers = result.x.reshape(-1, 2)
                    break # Found a valid one for this r, no need more restarts
        
        if min_penalty_found < 1e-9:
            r_low = r_mid # Feasible, try larger radius
        else:
            r_high = r_mid # Not feasible (or couldn't find solution), try smaller

    # Final result preparation
    if best_centers is None:
        # Fallback if optimization failed completely (unlikely)
        best_centers = np.zeros((n_circles, 2))
        best_r = 0.0

    radii = np.full(n_circles, best_r)
    sum_radii = np.sum(radii)
    
    return best_centers, radii, sum_radii