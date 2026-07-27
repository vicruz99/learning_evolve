import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    num_restarts = 10
    
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    # Helper to generate initial configurations
    def get_initial_guess(seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Strategy: 5x5 grid (25 circles) + 1 circle in a gap or random
        # Grid centers for 5x5
        step = 0.2
        centers = []
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                centers.append([step * (i + 0.5), step * (j + 0.5)])
        
        # Add 26th circle
        # Try to place it in the center of a "hole" or just random valid spot
        # A simple heuristic: place at (0.5, 0.5) but shift slightly to avoid overlap?
        # Actually (0.5, 0.5) is occupied by the center of the 5x5 grid (i=2, j=2).
        # Let's try placing it at a random valid location or a specific gap.
        # A gap in 5x5 grid is at (0.2, 0.2) relative to circle centers? 
        # Centers are at 0.1, 0.3, 0.5... Gaps at 0.2, 0.4...
        # Let's add a circle at (0.2, 0.2)
        centers.append([0.2, 0.2])
        
        centers = np.array(centers)
        
        # Add small random perturbation to break symmetry
        noise = np.random.uniform(-0.01, 0.01, size=centers.shape)
        centers = centers + noise
        # Clip to valid range
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial radii: small enough to not overlap
        radii = np.full(n, 0.04)
        
        return centers, radii

    # Objective function: maximize sum of radii => minimize -sum(radii)
    def objective(x_vec):
        # x_vec contains [x1, y1, r1, x2, y2, r2, ...]
        centers = x_vec[0::3].reshape(-1, 2)
        radii = x_vec[2::3] # Wait, indexing is tricky. 
        # Let's restructure x_vec: first n*2 are coords, next n are radii?
        # Or flat [x1, y1, r1, x2, y2, r2...]
        # Let's use separate arrays in closure or unpack carefully.
        # Here x_vec is flat.
        # Centers: x_vec[0:52], Radii: x_vec[52:78]?
        # Actually, let's just unpack.
        
        # Better structure for optimization:
        # variables = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        # So index i corresponds to circle i.
        # coords are at 3*i, 3*i+1. radius at 3*i+2.
        
        r = x_vec[2::3]
        return -np.sum(r)

    # Constraints
    # We need to define constraints for:
    # 1. x >= r
    # 2. 1 - x >= r
    # 3. y >= r
    # 4. 1 - y >= r
    # 5. dist(i, j) >= r_i + r_j
    
    # Defining hundreds of constraints explicitly in SLSQP is slow and memory heavy.
    # Instead, we can use a penalty method or handle constraints via bounds and a custom callback?
    # SLSQP supports non-linear constraints via a dict {'type': 'ineq', 'fun': ...}
    # But passing a list of constraint dicts is better.
    
    # However, constructing 325+ constraints is cumbersome.
    # Let's use a penalty method with L-BFGS-B or SLSQP with a single penalty constraint?
    # Or simply use 'trust-constr' which handles many constraints better?
    # 'trust-constr' is powerful but slower.
    
    # Let's try a penalty approach inside the objective, optimizing with L-BFGS-B which supports bounds.
    # Bounds: r_i >= 0. x, y in [0, 1].
    
    # To make it robust, we will optimize:
    # f(vars) = -sum(r) + Penalty
    # Penalty is active only when constraints are violated.
    
    # Let's restructure the variable vector to be flat [x1, y1, x2, y2, ..., r1, r2, ...]
    # No, let's keep [x1, y1, r1, ...] for easier indexing, but bounds are mixed.
    # L-BFGS-B bounds are box constraints. 
    # We can bound r >= 0, x, y in [0, 1].
    # The complex constraints (overlap) must be penalized in objective.
    
    # Variable vector: size 3*n.
    # Order: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    
    def objective_penalty(x_vec):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = x_vec[3*i]
            centers[i, 1] = x_vec[3*i+1]
            radii[i] = x_vec[3*i+2]
            
        obj = -np.sum(radii)
        penalty = 0.0
        
        # Boundary penalties
        # x - r >= 0  => r - x <= 0
        # 1 - x - r >= 0 => x + r - 1 <= 0
        # y - r >= 0
        # 1 - y - r >= 0
        
        # Using a large weight for penalties to force feasibility
        # But penalty should be squared violation to be smooth
        
        # Boundary
        for i in range(n):
            r = radii[i]
            x = centers[i, 0]
            y = centers[i, 1]
            
            # Left wall
            if x - r < 0:
                penalty += (x - r)**2
            # Right wall
            if x + r > 1:
                penalty += (x + r - 1)**2
            # Bottom wall
            if y - r < 0:
                penalty += (y - r)**2
            # Top wall
            if y + r > 1:
                penalty += (y + r - 1)**2
                
        # Overlap penalties
        # dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
        # violation if dist < r_i + r_j
        
        # Vectorized distance calculation
        # Centers shape (n, 2)
        # diff = centers[:, None, :] - centers[None, :, :] -> (n, n, 2)
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # radii sum matrix: r[:, None] + r[None, :]
        
        # Optimization of this loop:
        # Only check i < j
        
        # Let's do a simple loop, n=26 is small
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = radii[i] + radii[j]
                if dist < r_sum:
                    penalty += (r_sum - dist)**2
                    
        # Weight for penalty. 
        # If sum of radii is around 2.6, and violations are small, we need high weight.
        # Let's start with 1000.
        weight = 1000.0
        return obj + weight * penalty

    # Bounds: x, y in [0, 1], r in [0, 0.5] (max possible radius is 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    best_state = None

    for k in range(num_restarts):
        # Initial guess
        # Use a mix of grid and random
        centers_init, radii_init = get_initial_guess(seed=k)
        
        # Flatten to 1D vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        # Optimization
        # L-BFGS-B handles bounds well.
        res = minimize(objective_penalty, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-9})
        
        # Check validity of result
        # Unpack
        centers_res = np.zeros((n, 2))
        radii_res = np.zeros(n)
        for i in range(n):
            centers_res[i, 0] = res.x[3*i]
            centers_res[i, 1] = res.x[3*i+1]
            radii_res[i] = res.x[3*i+2]
            
        # Verify constraints manually (strictly) to filter invalid solutions
        valid = True
        for i in range(n):
            x, y = centers_res[i]
            r = radii_res[i]
            if x - r < -1e-5 or x + r > 1 + 1e-5 or y - r < -1e-5 or y + r > 1 + 1e-5:
                valid = False
                break
        
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((centers_res[i,0]-centers_res[j,0])**2 + (centers_res[i,1]-centers_res[j,1])**2)
                    if dist < radii_res[i] + radii_res[j] - 1e-5:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            current_sum = np.sum(radii_res)
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_centers = centers_res.copy()
                best_radii = radii_res.copy()
                best_state = res
        
        # Print progress
        # print(f"Restart {k}: Sum radii = {np.sum(radii_res):.5f}, Valid = {valid}")

    # If no valid solution found (unlikely with good init), return a fallback
    if best_centers is None:
        # Fallback: 5x5 grid with small radius
        centers_fallback = np.zeros((25, 2))
        for i in range(5):
            for j in range(5):
                centers_fallback[5*i+j] = [0.1 + 0.2*i, 0.1 + 0.2*j]
        # Add 26th circle
        centers_full = np.vstack([centers_fallback, [[0.5, 0.5]]]) # Wait, 0.5,0.5 is occupied.
        # Just place it somewhere safe, e.g., (0.95, 0.95)
        centers_full = np.vstack([centers_fallback, [[0.95, 0.95]]])
        radii_fallback = np.full(26, 0.05)
        return centers_full, radii_fallback, np.sum(radii_fallback)

    return best_centers, best_radii, best_sum_radii