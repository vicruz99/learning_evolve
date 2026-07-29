# sol_000350 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a8bfd9ed) state=e2351a8c sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Strategy:
    1. Initialize centers in a hexagonal grid pattern with a safe radius.
    2. Use scipy.optimize.minimize (SLSQP) to maximize the sum of radii
       subject to boundary and non-overlap constraints.
    """
    n = 26
    
    # --- 1. Initialization ---
    # Start with a hexagonal grid packing. 
    # Radius 0.08 is chosen to be small enough to fit easily but large enough to be close to optimal.
    r_init = 0.08
    centers = []
    
    y = r_init
    row = 0
    while len(centers) < n:
        # Hexagonal staggering: even rows start at r, odd rows start at 2r (shifted by r)
        # x spacing is 2r (diameter)
        if row % 2 == 0:
            x = r_init
        else:
            x = 2 * r_init 
        
        while x + r_init <= 1.0 + 1e-9: # Ensure circle stays inside right boundary
            centers.append([x, y])
            if len(centers) >= n:
                break
            x += 2 * r_init
        
        y += np.sqrt(3) * r_init
        if y + r_init > 1.0 + 1e-9:
            break
        row += 1
    
    # If for some reason we didn't get enough (unlikely with r=0.08), pad with random points
    if len(centers) < n:
        while len(centers) < n:
            centers.append([np.random.rand(), np.random.rand()])
            
    centers = np.array(centers[:n])
    
    # Add small random noise to break symmetry and help the optimizer
    centers += np.random.uniform(-0.0001, 0.0001, centers.shape)
    
    # --- 2. Optimization Setup ---
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    x0 = []
    for i in range(n):
        x0.extend([centers[i, 0], centers[i, 1], r_init])
    x0 = np.array(x0)
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
    
    # Define objective: Minimize -sum(radii)
    def objective(v):
        # Radii are at indices 2, 5, 8, ...
        return -np.sum(v[2::3])

    # Define constraints function (Vectorized for performance)
    # Returns an array where every element must be >= 0
    def constraints_func(v):
        # Extract arrays
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        cons = []
        
        # 1. Boundary Constraints
        # x - r >= 0
        cons.append(xs - rs)
        # 1 - x - r >= 0
        cons.append(1.0 - xs - rs)
        # y - r >= 0
        cons.append(ys - rs)
        # 1 - y - r >= 0
        cons.append(1.0 - ys - rs)
        
        # 2. Pairwise Non-overlap Constraints
        # dist^2 >= (r_i + r_j)^2  =>  dist^2 - (r_i + r_j)^2 >= 0
        # Use broadcasting for all pairs
        # Shapes: (n, 1) and (1, n)
        dx = xs[:, None] - xs[None, :] 
        dy = ys[:, None] - ys[None, :]
        dr = rs[:, None] + rs[None, :]
        
        dist_sq = dx**2 + dy**2
        sum_r_sq = dr**2
        
        # We only need upper triangle (i < j) to avoid duplicates and self-checks
        # Create a mask for upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pair_cons = (dist_sq - sum_r_sq)[mask]
        
        cons.append(pair_cons)
        
        return np.concatenate(cons)

    # Set up constraint dictionary for scipy
    # 'ineq' means the function value should be >= 0
    cons = {'type': 'ineq', 'fun': constraints_func}

    # --- 3. Run Optimization ---
    # SLSQP is a sequential least squares programming algorithm suitable for this
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={
            'ftol': 1e-10, 
            'maxiter': 2000, 
            'disp': False
        }
    )
    
    # --- 4. Extract Results ---
    final_centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
    final_radii = res.x[2::3]
    sum_radii = float(np.sum(final_radii))
    
    return final_centers, final_radii, sum_radii

# Helper to run and print results if executed directly
if __name__ == "__main__":
    # Dummy validate function to simulate the check
    def validate_packing(centers, radii):
        n = centers.shape[0]
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if r < 0: return False
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    print(f"Valid: {validate_packing(centers, radii)}")
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")
