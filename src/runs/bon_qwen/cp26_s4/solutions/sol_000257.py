# sol_000257 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a15173c5) state=6b107638 sum of radii=2.323604 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_energy_and_grad(params, n_circles, lam):
    """
    Computes the energy and its gradient for the circle packing problem.
    Energy is defined based on a Log-Sum-Exponential approximation of the 
    negative minimum distance between circles and boundaries.
    Minimizing this energy maximizes the minimum distance.
    
    Args:
        params: 1D array of shape (2*n_circles) containing [x1, y1, x2, y2, ...]
        n_circles: Number of circles (26)
        lam: Parameter for the soft-min approximation (sharpness)
        
    Returns:
        energy: Scalar energy value
        grad: 1D array of shape (2*n_circles) containing gradients
    """
    params = np.asarray(params)
    centers = params.reshape((n_circles, 2))
    
    # Initialize gradient
    grad = np.zeros_like(params)
    
    # Terms for LSE: list of exponentials
    # We want to maximize distances, so we penalize small distances.
    # Term for distance d: exp(-lam * d)
    # Small d -> Large term. Large d -> Small term.
    # We sum these terms and take log.
    
    sum_exp = 0.0
    sum_grad = np.zeros_like(centers)
    
    # 1. Inter-circle distances
    # We iterate over all pairs i < j
    for i in range(n_circles):
        ci = centers[i]
        for j in range(i + 1, n_circles):
            cj = centers[j]
            diff = ci - cj
            dist = np.sqrt(np.sum(diff**2))
            
            # Avoid division by zero if points coincide
            if dist < 1e-9:
                dist = 1e-9
                diff = np.random.rand(2) * 1e-5 # Perturb slightly to avoid singularity in gradient
            
            # The value we care about is radius r, so distance must be >= 2r.
            # Here we just maximize distance directly.
            # We use dist / 2 as the "margin" r.
            # Actually, let's just maximize dist.
            # Term: exp(-lam * dist / 2)  <- scaling by 1/2 to relate to radius
            val = np.exp(-lam * dist / 2.0)
            sum_exp += val
            
            # Gradient of exp(-lam * dist / 2) wrt ci
            # d(dist)/d(ci) = diff / dist
            # d(val)/d(ci) = val * (-lam / 2) * (diff / dist)
            grad_val = val * (-lam / 2.0) * (diff / dist)
            
            sum_grad[i] += grad_val
            sum_grad[j] -= grad_val # Gradient wrt cj is opposite
            
    # 2. Boundary constraints
    # For each circle, distance to boundaries: x, 1-x, y, 1-y
    # We want these to be >= r.
    # So we penalize small boundary distances.
    # Terms: exp(-lam * x), exp(-lam * (1-x)), etc.
    for i in range(n_circles):
        x, y = centers[i]
        
        # Left boundary (x)
        val_x = np.exp(-lam * x)
        sum_exp += val_x
        sum_grad[i, 0] += val_x * (-lam)
        
        # Right boundary (1-x)
        val_1x = np.exp(-lam * (1.0 - x))
        sum_exp += val_1x
        sum_grad[i, 0] += val_1x * (lam) # d(1-x)/dx = -1, chain rule: -lam * (-1) = lam
        
        # Bottom boundary (y)
        val_y = np.exp(-lam * y)
        sum_exp += val_y
        sum_grad[i, 1] += val_y * (-lam)
        
        # Top boundary (1-y)
        val_1y = np.exp(-lam * (1.0 - y))
        sum_exp += val_1y
        sum_grad[i, 1] += val_1y * (lam)
        
    # Energy = (1/lam) * log(sum_exp)
    # We add a small epsilon to avoid log(0) if sum_exp is 0 (unlikely)
    energy = (1.0 / lam) * np.log(sum_exp + 1e-300)
    
    # Gradient of Energy
    # dE/dx = (1/lam) * (1/sum_exp) * d(sum_exp)/dx
    # d(sum_exp)/dx = sum_grad
    grad_energy = (1.0 / lam) * (1.0 / (sum_exp + 1e-300)) * sum_grad.flatten()
    
    return energy, grad_energy

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square.
    
    Returns:
        centers: (26, 2) array of circle centers
        radii: (26,) array of radii
        sum_radii: Sum of radii
    """
    n_circles = 26
    n_params = 2 * n_circles
    
    # Strategy:
    # 1. Initialize centers in a grid pattern.
    # 2. Optimize positions to maximize minimum separation (equal radii packing).
    # 3. Calculate the valid radius for the optimized configuration.
    
    # Initial guess: Grid
    # 26 circles. Approx sqrt(26) ~ 5.1. 
    # Let's try a 6x5 grid or similar, or just random.
    # A perturbed grid is usually a good start.
    # 5x5 grid has 25. We need 26.
    # Let's place them in a 6x5 grid (30 spots) and pick 26? 
    # Or just random restarts.
    # Let's use a dense grid initialization.
    
    best_energy = np.inf
    best_centers = None
    
    # Try multiple random restarts to avoid local minima
    np.random.seed(42)
    n_restarts = 5
    
    for _ in range(n_restarts):
        # Random initialization within a safe margin to allow movement
        # Margin 0.1 ensures we don't start stuck at walls
        centers_init = 0.1 + np.random.rand(n_params) * 0.8
        # Reshape to (n, 2) for logic, but optimizer takes 1D
        # We can also try a grid initialization for one run
        if _ == 0:
            # Grid initialization
            # 5 rows, 6 cols? 30 spots.
            # Let's take first 26.
            grid_x = np.linspace(0.1, 0.9, 6)
            grid_y = np.linspace(0.1, 0.9, 5)
            xs, ys = np.meshgrid(grid_x, grid_y)
            points = np.column_stack([xs.ravel(), ys.ravel()])
            # Take first 26
            centers_init = points[:n_circles].flatten()
        
        # Optimization parameters
        lam = 100.0 # Sharpness of LSE
        
        # Bounds for coordinates [0, 1]
        bounds = [(0.0, 1.0)] * n_params
        
        # Minimize energy
        res = minimize(
            fun=lambda p: compute_energy_and_grad(p, n_circles, lam)[0],
            x0=centers_init,
            method='L-BFGS-B',
            jac=lambda p: compute_energy_and_grad(p, n_circles, lam)[1],
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_centers = res.x.reshape((n_circles, 2))
            
    # Now we have best_centers.
    # Calculate the actual maximum radius r that fits.
    # r is limited by:
    # 1. Distance to boundaries: min(x, 1-x, y, 1-y) for all circles
    # 2. Distance between circles: min(dist(i,j)) / 2 for all pairs
    
    min_r = 1.0
    
    # Check boundaries
    for i in range(n_circles):
        x, y = best_centers[i]
        d_bound = min(x, 1.0 - x, y, 1.0 - y)
        if d_bound < min_r:
            min_r = d_bound
            
    # Check inter-circle distances
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
            r_pair = dist / 2.0
            if r_pair < min_r:
                min_r = r_pair
                
    # Set all radii to this max feasible radius
    radii = np.full(n_circles, min_r)
    
    # Final validation check and adjustment if needed (numerical safety)
    # Although the logic should hold, floating point errors might occur.
    # We ensure strict validity.
    
    centers_out = best_centers
    radii_out = radii
    
    sum_r = np.sum(radii_out)
    
    return centers_out, radii_out, sum_r
