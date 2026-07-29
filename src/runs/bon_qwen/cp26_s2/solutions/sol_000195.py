# sol_000195 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fb76805b) state=4551f7fd sum of radii=2.426549 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n):
    """
    Computes the negative sum of radii plus penalties for boundary and overlap violations.
    """
    C = vars[:2*n].reshape(n, 2)
    R = vars[2*n:]
    
    # Boundary penalty: circles must stay within [0,1]x[0,1]
    p1 = np.maximum(R - C[:, 0], 0)
    p2 = np.maximum(C[:, 0] + R - 1, 0)
    p3 = np.maximum(R - C[:, 1], 0)
    p4 = np.maximum(C[:, 1] + R - 1, 0)
    pen_bound = np.sum(p1**2 + p2**2 + p3**2 + p4**2)
    
    # Overlap penalty: distance between centers must be >= sum of radii
    dx = C[:, None, :] - C[None, :, :]
    dist = np.sqrt(np.sum(dx**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    overlap = R[:, None] + R[None, :] - dist
    pen_overlap = np.sum(np.maximum(overlap, 0)**2)
    
    # Objective: maximize sum of radii -> minimize negative sum
    return -np.sum(R) + 2000.0 * (pen_bound + pen_overlap)

def run_packing():
    np.random.seed(42)
    n = 26
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Run multiple trials with different initial configurations to avoid poor local minima
    for trial in range(3):
        init_centers = []
        # Start with a hexagonal packing pattern, slightly varying initial radius
        r_init = 0.09 + trial * 0.005
        s = 2 * r_init
        y = r_init
        row = 0
        while len(init_centers) < n and y + r_init <= 1.0:
            x = r_init + (s/2 if row % 2 == 1 else 0)
            while x + r_init <= 1.0 and len(init_centers) < n:
                init_centers.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
            
        # Fill remaining spots with random positions if hex pattern doesn't yield 26
        while len(init_centers) < n:
            init_centers.append([np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1])
            
        init_centers = np.array(init_centers[:n])
        init_radii = np.full(n, r_init)
        
        # Add small random perturbation to escape grid symmetry
        init_centers += np.random.randn(*init_centers.shape) * 0.005
        init_centers = np.clip(init_centers, 0.02, 0.98)
        
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        # Optimize using L-BFGS-B with penalty method
        res = minimize(compute_objective, x0, args=(n,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-12})
        
        cur_radii = res.x[2*n:]
        actual_sum = np.sum(cur_radii)
        
        if actual_sum > best_sum:
            best_sum = actual_sum
            best_centers = res.x[:2*n].reshape(n, 2).copy()
            best_radii = cur_radii.copy()
            
    # Post-processing: ensure strict validity by conservatively adjusting radii
    # This guarantees the validation function passes without numerical edge cases
    for i in range(n):
        x, y = best_centers[i]
        limit = min(x, 1-x, y, 1-y)
        for j in range(n):
            if i != j:
                dist = np.sqrt((x - best_centers[j,0])**2 + (y - best_centers[j,1])**2)
                limit = min(limit, dist - best_radii[j])
        best_radii[i] = max(0.0, limit - 1e-9)
        
    return best_centers, best_radii, np.sum(best_radii)
