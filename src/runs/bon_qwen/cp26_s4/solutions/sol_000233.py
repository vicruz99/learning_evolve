# sol_000233 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2fe8b400) state=1be30fa3 sum of radii=2.591232 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    N = 26
    
    # 1. Initialization: 6x5 grid
    # This provides a good spread of centers to start with
    centers = np.zeros((N, 2))
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < N:
                # Place centers evenly in (0.1, 0.9) range roughly
                x = 0.5/6 + i * (0.9)/6 # Actually just linspace
                y = 0.5/5 + j * (0.9)/5
                # Let's use a proper linspace for better coverage
                centers[idx] = [0.5/6 + i * (1 - 1/6) / 5, 0.5/5 + j * (1 - 1/5) / 4] # Rough guess
                # Better:
                centers[idx, 0] = (i + 0.5) / 6
                centers[idx, 1] = (j + 0.5) / 5
                idx += 1
    
    # Precompute pair indices for LP constraints
    pair_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_indices.append((i, j))
    n_pairs = len(pair_indices)
    
    # Boundary constraints indices (4 per circle)
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    
    # Optimization parameters
    max_iter = 300
    step_size = 0.05
    
    for iteration in range(max_iter):
        # 2. Solve LP for radii given fixed centers
        # Variables: r_0 ... r_{N-1}
        # Objective: Max sum(r_i) -> Min -sum(r_i)
        c_obj = np.ones(N) # We minimize, so -1 sum is max sum. Wait, linprog minimizes c^T x.
        # So we set c = -1.
        c_lp = -np.ones(N)
        
        # Constraints: A_ub @ r <= b_ub
        # Pair constraints: r_i + r_j <= dist(i, j)
        # Boundary constraints: r_i <= bound_i
        
        n_constraints = n_pairs + 4 * N
        A_ub = np.zeros((n_constraints, N))
        b_ub = np.zeros(n_constraints)
        
        # Fill pair constraints
        for k, (i, j) in enumerate(pair_indices):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dist
            
        # Fill boundary constraints
        # r_i <= x_i
        # r_i <= 1-x_i
        # r_i <= y_i
        # r_i <= 1-y_i
        for i in range(N):
            base_idx = n_pairs + 4 * i
            x, y = centers[i]
            
            A_ub[base_idx, i] = 1.0
            b_ub[base_idx] = x
            
            A_ub[base_idx+1, i] = 1.0
            b_ub[base_idx+1] = 1.0 - x
            
            A_ub[base_idx+2, i] = 1.0
            b_ub[base_idx+2] = y
            
            A_ub[base_idx+3, i] = 1.0
            b_ub[base_idx+3] = 1.0 - y
            
        # Bounds for radii: r_i >= 0
        bounds = [(0, None) for _ in range(N)]
        
        # Solve LP
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # If LP fails, try to continue or break
            # Usually shouldn't fail with valid geometry
            pass
            
        radii = res.x
        current_sum = np.sum(radii)
        
        # 3. Calculate forces and update centers
        # We want to move centers to increase the RHS of active constraints.
        # Active pair constraint: r_i + r_j approx dist(i, j).
        # Active boundary constraint: r_i approx bound.
        
        forces = np.zeros((N, 2))
        
        # Pair forces
        for k, (i, j) in enumerate(pair_indices):
            r_i, r_j = radii[i], radii[j]
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            
            # Check if constraint is active (tight)
            # Allow small tolerance for numerical stability
            if dist < 1e-9: dist = 1e-9 # Avoid division by zero
            
            slack = dist - (r_i + r_j)
            
            if slack < 1e-4: # Tight constraint
                # Direction to increase distance: vector from j to i
                diff = centers[i] - centers[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-9:
                    dir_vec = diff / norm
                    # Repulsion force
                    # Strength could be proportional to how tight it is or just constant
                    # Let's use a repulsion that pushes them apart
                    repulsion_strength = 1.0 
                    forces[i] += repulsion_strength * dir_vec
                    forces[j] -= repulsion_strength * dir_vec
        
        # Boundary forces
        for i in range(N):
            r_i = radii[i]
            x, y = centers[i]
            
            # Left: x - r_i approx 0 => r_i approx x. Push right (+x)
            if x - r_i < 1e-4:
                forces[i, 0] += 1.0
            # Right: 1 - x - r_i approx 0 => r_i approx 1-x. Push left (-x)
            if (1.0 - x) - r_i < 1e-4:
                forces[i, 0] -= 1.0
            # Bottom: y - r_i approx 0 => r_i approx y. Push up (+y)
            if y - r_i < 1e-4:
                forces[i, 1] += 1.0
            # Top: 1 - y - r_i approx 0 => r_i approx 1-y. Push down (-y)
            if (1.0 - y) - r_i < 1e-4:
                forces[i, 1] -= 1.0
        
        # Update centers
        # Decay step size
        current_step = step_size / (1 + iteration * 0.1)
        
        centers = centers + current_step * forces
        
        # Clip centers to valid range [0, 1]
        # Actually centers must be within [0, 1], but strictly speaking 
        # if r > 0, center must be in (0, 1). 
        # Clipping to [0, 1] is safe.
        centers = np.clip(centers, 0, 1)
        
        # Occasionally add noise to escape local minima?
        # Maybe not needed if gradient is strong enough.
        # But let's add tiny noise in last iterations?
        # No, let's rely on forces.
        
    # Final LP solve to get precise radii for the optimized centers
    # Re-run the LP setup with final centers
    A_ub = np.zeros((n_constraints, N))
    b_ub = np.zeros(n_constraints)
    
    for k, (i, j) in enumerate(pair_indices):
        dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = dist
        
    for i in range(N):
        base_idx = n_pairs + 4 * i
        x, y = centers[i]
        A_ub[base_idx, i] = 1.0
        b_ub[base_idx] = x
        A_ub[base_idx+1, i] = 1.0
        b_ub[base_idx+1] = 1.0 - x
        A_ub[base_idx+2, i] = 1.0
        b_ub[base_idx+2] = y
        A_ub[base_idx+3, i] = 1.0
        b_ub[base_idx+3] = 1.0 - y
        
    bounds = [(0, None) for _ in range(N)]
    res_final = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    final_radii = res_final.x
    final_sum = np.sum(final_radii)
    
    # Verify and adjust if necessary (numerical errors)
    # Ensure radii are valid (non-negative, inside box)
    # The LP should handle this, but let's clamp just in case
    final_radii = np.maximum(final_radii, 0)
    
    # Check if any circle is outside due to radius
    # Actually LP ensures r <= x etc.
    
    return centers, final_radii, final_sum
