# sol_000172 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ae65bcc8) state=0ed92758 sum of radii=2.364417 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_max_radii(centers):
    """Solve LP to find maximum radii for fixed center positions."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    
    # Upper bounds from square boundaries
    ub = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    ub = np.maximum(ub, 0.0)
    bounds = [(0.0, b) for b in ub]
    
    # Pairwise non-overlap constraints: r_i + r_j <= dist(i,j)
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    k = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            dist = np.hypot(xi - x[j], yi - y[j])
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dist
            k += 1
            
    try:
        # 'highs' is a modern, robust LP solver available in recent scipy versions
        res = linprog(c=-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                      method='highs', options={'disp': False})
        if res.success:
            return res.x
    except Exception:
        pass
        
    # Fallback: small radii to avoid NaNs
    return np.full(n, 1e-7)

def objective(centers_flat):
    """Negative sum of radii to be minimized."""
    centers = centers_flat.reshape(-1, 2)
    # Keep centers strictly inside to maintain positive radii bounds
    centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
    radii = compute_max_radii(centers)
    return -np.sum(radii)

def get_initial_guesses(n):
    """Generate diverse starting configurations for centers."""
    guesses = []
    np.random.seed(42)
    
    # 1. Hexagonal lattice pattern (high packing density baseline)
    pts = []
    s = 0.18  # Approximate spacing
    for row in range(12):
        y = row * s * np.sqrt(3) / 2
        shift = (row % 2) * s / 2
        for col in range(8):
            x = col * s + shift
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
        guesses.append(pts[idx].flatten())
        
    # 2. Uniform random distribution
    guesses.append(np.random.rand(n * 2))
    
    # 3. Structured grid
    gs = np.linspace(0.1, 0.9, 6)
    grid = np.array([[x, y] for y in gs for x in gs])
    idx = np.random.choice(len(grid), n, replace=False)
    guesses.append(grid[idx].flatten())
    
    return guesses

def run_packing():
    """Optimize circle packing to maximize sum of radii."""
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    initials = get_initial_guesses(26)
    
    # Run local optimization from each initial guess
    for init in initials:
        res = minimize(objective, init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'adaptive': True, 
                                'xatol': 1e-6, 'fatol': 1e-9})
        val = -res.fun
        if val > best_sum:
            best_sum = val
            best_centers = res.x.reshape(-1, 2)
            
    if best_centers is None:
        best_centers = np.random.rand(26, 2) * 0.8 + 0.1
        
    # Compute final optimal radii
    best_radii = compute_max_radii(best_centers)
    best_centers = np.clip(best_centers, 0, 1)
    
    return best_centers, best_radii, best_sum
