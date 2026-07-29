# sol_000301 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 525683f8) state=d35a61ce sum of radii=0.006500 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We generate points in a hexagonal pattern and select n points that fit well
    # or simply create a dense grid and trim/shift.
    # A hexagonal grid with spacing ~0.2 allows roughly 26 circles in the unit square.
    
    centers = []
    r_init = 0.1
    dy = r_init * np.sqrt(3)  # Vertical distance between rows
    dx = r_init * 2.0         # Horizontal distance between circles
    
    y = 0.0
    row = 0
    while len(centers) < n and y <= 1.0:
        x = 0.0 if row % 2 == 0 else dx / 2.0
        col = 0
        while x <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += dx
            col += 1
        y += dy
        row += 1
    
    # Normalize centers to be well within the box (add small margin)
    # The generated centers might be near 0 or 1, radii will handle the exact fit
    centers = np.array(centers[:n])
    # Shift to center if necessary, but keeping raw grid is usually fine for local search
    # Let's add a bit of random noise to break symmetry and avoid local minima
    np.random.seed(123)
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, 0.01, 0.99)

    # 2. Optimization Setup
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    init_vars = np.zeros(n * 3)
    for i in range(n):
        init_vars[3*i] = centers[i, 0]
        init_vars[3*i+1] = centers[i, 1]
        init_vars[3*i+2] = 0.05  # Initial radius guess
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    def objective(vars):
        # Reshape variables
        pos = vars.reshape(n, 3)
        centers = pos[:, :2]
        radii = pos[:, 2]
        
        # Primary Objective: Maximize sum of radii (minimize negative sum)
        obj = -np.sum(radii)
        
        # Penalty Constants
        penalty_weight = 500.0 
        
        # 1. Boundary Penalties
        # Constraint: r <= x <= 1-r  =>  x - r >= 0,  1 - x - r >= 0
        # Constraint: r <= y <= 1-r  =>  y - r >= 0,  1 - y - r >= 0
        # We penalize the square of the violation amount
        for i in range(n):
            x, y, r = pos[i]
            
            # Left boundary
            if x - r < 0:
                obj += penalty_weight * (x - r)**2
            # Right boundary
            if x + r > 1:
                obj += penalty_weight * (x + r - 1)**2
            # Bottom boundary
            if y - r < 0:
                obj += penalty_weight * (y - r)**2
            # Top boundary
            if y + r > 1:
                obj += penalty_weight * (y + r - 1)**2
            # Negative radius
            if r < 0:
                obj += penalty_weight * r**2

        # 2. Overlap Penalties
        # Constraint: dist(i, j) >= r_i + r_j
        # Violation: (r_i + r_j) - dist(i, j) > 0
        # We use a smooth penalty function
        
        # Vectorized distance calculation for performance
        # Compute all pairwise distances
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Radii sum matrix
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # We only need upper triangle to avoid double counting, but full matrix is easier to vectorize
        # Violation matrix
        violation = r_sum - dists
        
        # We only care where violation > 0
        # To make it smooth, we can use a function that approximates max(0, x)^2
        # or just square the positive parts.
        # Since we are using gradient-based optimizer, max(0, x)^2 is continuous but derivative is 0 at 0.
        # It's generally acceptable.
        
        # Mask for violations
        mask = violation > 0
        penalty_sum = np.sum(mask * (violation**2))
        
        obj += penalty_weight * penalty_sum
        
        return obj

    # Run Optimization
    # L-BFGS-B is good for bounds and large scale
    result = minimize(
        objective, 
        init_vars, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={
            'maxiter': 2000, 
            'ftol': 1e-12,
            'gtol': 1e-10
        }
    )
    
    # Extract results
    final_vars = result.x
    final_pos = final_vars.reshape(n, 3)
    final_centers = final_pos[:, :2]
    final_radii = final_pos[:, 2]
    
    # Clean up radii (ensure non-negative, though bounds should handle it)
    final_radii = np.maximum(final_radii, 0)
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

if __name__ == "__main__":
    # Validation test
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    
    # Check validity manually for debugging
    # (Note: The actual validation function provided in prompt is used by the judge)
    n = centers.shape[0]
    valid = True
    
    # Check bounds and negative radii
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0:
            valid = False
            print(f"Circle {i} has negative radius")
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            print(f"Circle {i} outside boundary")

    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                print(f"Circles {i} and {j} overlap")
    
    if valid:
        print("Packing is VALID.")
    else:
        print("Packing is INVALID.")
