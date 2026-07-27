# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22281c24) state=97b6c401 sum of radii=2.590170 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # Helper function to convert optimization variables to physical centers and radii
    # Variables order: [u0, v0, r0, u1, v1, r1, ...]
    def get_state(vars):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            u = vars[3*i]
            v = vars[3*i+1]
            r = vars[3*i+2]
            
            # Bounds on r are handled by optimizer, but clamp just in case
            r = max(0.0, min(0.5, r))
            
            # Transformation to ensure boundary constraints
            # x = r + u * (1 - 2r) maps u in [0,1] to x in [r, 1-r]
            # If r=0.5, x=0.5. If r=0, x=u.
            x = r + u * (1.0 - 2.0 * r)
            y = r + v * (1.0 - 2.0 * r)
            
            centers[i] = (x, y)
            radii[i] = r
        return centers, radii

    # Objective function: minimize negative sum of radii
    def objective(vars):
        centers, radii = get_state(vars)
        return -np.sum(radii)

    # Constraints: non-overlap
    # dist(C_i, C_j) >= r_i + r_j
    # dist - (r_i + r_j) >= 0
    def constraint_non_overlap(vars):
        centers, radii = get_state(vars)
        
        # Vectorized distance calculation
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Sum of radii matrix
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Constraint value: dists - r_sum
        # We only need to return the upper triangular part (excluding diagonal)
        # to avoid redundant constraints and self-overlap checks (diagonal is 0 - 2r <= 0 which is bad)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        constraint_values = dists[mask] - r_sum[mask]
        
        return constraint_values

    # Bounds for variables
    # u, v in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # u
        bounds.append((0.0, 1.0)) # v
        bounds.append((0.0, 0.5)) # r

    # Define constraints structure for SLSQP
    cons = {
        'type': 'ineq',
        'fun': constraint_non_overlap
    }

    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None

    # Multi-start optimization
    # We try different initial configurations to escape local minima
    num_starts = 10
    
    for seed in range(num_starts):
        rng = np.random.RandomState(seed)
        
        # Strategy for initialization:
        # 1. Random uniform
        # 2. Grid-based with perturbation (often better for packing)
        
        vars_init = np.zeros(3 * n)
        
        # Try to pack in a dense grid initially to give optimizer a head start
        # 26 circles. Approx grid size 5x5 or 6x5?
        # Let's try a hexagonal-ish layout or just random if grid is tricky to fit exactly.
        # Random is safer to avoid bad initial overlaps that might trap gradient.
        
        # Let's use a jittered grid for better coverage
        cols = 6
        rows = 5 # 30 spots, we pick 26
        # Or just fill 26 points randomly
        # Let's do random first, then maybe a grid try in later seeds
        
        if seed < 5:
            # Random initialization
            for i in range(n):
                vars_init[3*i] = rng.rand()      # u
                vars_init[3*i+1] = rng.rand()    # v
                vars_init[3*i+2] = 0.02          # r (small start)
        else:
            # Grid initialization
            # Place centers roughly in a grid, convert to u, v
            # Grid spacing approx 1/6
            step = 1.0 / 6.0
            count = 0
            r_init = 0.05 # Slightly larger guess
            
            # Create a list of grid points
            points = []
            for r in range(6):
                for c in range(6):
                    if count < n:
                        # Center slightly offset to avoid boundaries
                        x = (c + 0.5) * (1.0 / 6.0) + rng.rand() * 0.05 - 0.025
                        y = (r + 0.5) * (1.0 / 6.0) + rng.rand() * 0.05 - 0.025
                        
                        # Clip to [0, 1]
                        x = np.clip(x, 0.0, 1.0)
                        y = np.clip(y, 0.0, 1.0)
                        
                        # Convert back to u, v given r_init
                        # x = r + u(1-2r) => u = (x - r) / (1 - 2r)
                        if 1 - 2*r_init > 1e-9:
                            u = (x - r_init) / (1 - 2*r_init)
                            v = (y - r_init) / (1 - 2*r_init)
                        else:
                            u = 0.5
                            v = 0.5
                            
                        # Clip u, v to [0, 1]
                        u = np.clip(u, 0.0, 1.0)
                        v = np.clip(v, 0.0, 1.0)
                        
                        vars_init[3*count] = u
                        vars_init[3*count+1] = v
                        vars_init[3*count+2] = r_init
                        count += 1
        
        # Run optimization
        try:
            res = minimize(
                objective, 
                vars_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or (not np.isinf(res.fun) and not np.isnan(res.fun)):
                centers, radii = get_state(res.x)
                current_sum = np.sum(radii)
                
                # Validate the solution strictly (handling numerical errors)
                # Check overlaps
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                        if dist < radii[i] + radii[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = centers.copy()
                        best_radii = radii.copy()
                        
        except Exception:
            continue

    # If no valid solution found (unlikely), return a fallback
    if best_centers is None:
        # Fallback: 26 tiny circles
        best_centers = np.random.rand(26, 2)
        best_radii = np.full(26, 0.01)
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii
