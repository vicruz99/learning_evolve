# sol_000323 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ef4a4e64) state=689eb16d sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Geometric Initialization (Staggered Hexagonal Grid)
    # Attempt to pack 5 rows of 5 circles and 1 row of 1 circle (26 total)
    # We use a slightly compressed grid to start, allowing the solver to expand.
    centers = np.zeros((n, 2))
    r_init = 0.05
    row, col = 0, 0
    for i in range(n):
        centers[i, 0] = r_init + col * (2 * r_init) + (0.5 * r_init if row % 2 == 1 else 0)
        centers[i, 1] = r_init + row * (np.sqrt(3) * r_init)
        
        col += 1
        # Alternate row lengths for hexagonal pattern
        if row % 2 == 0 and col >= 5:
            col = 0
            row += 1
        elif row % 2 == 1 and col >= 4:
            col = 0
            row += 1
            
    # Ensure all points are within [0, 1]
    centers = np.clip(centers, 0.01, 0.99)

    def solve_radii(centers):
        n = centers.shape[0]
        c = np.ones(n) # Maximize sum r_i
        
        # Constraint matrix A_ub @ vars <= b_ub
        # Constraints: r_i + r_j <= dist_ij
        # Also r_i <= boundary_dist
        
        A_ub = []
        b_ub = []
        
        # Distance constraints between circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dists[i, j] = np.linalg.norm(centers[i] - centers[j])
        
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
                
        # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            x, y = centers[i]
            bd = min(x, 1-x, y, 1-y)
            A_ub.append(row)
            b_ub.append(bd)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for radii [0, 0.5]
        bounds = [(0, 0.5) for _ in range(n)]
        
        # Solve LP
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            # Fallback if LP fails (e.g., infeasible), return small radii
            return np.full(n, 0.01)

    # 2. Iterative Optimization
    for _ in range(500): # 500 iterations
        radii = solve_radii(centers)
        
        # Calculate repulsive forces based on active constraints
        forces = np.zeros_like(centers)
        
        for i in range(n):
            # Boundary forces
            x, y = centers[i]
            r = radii[i]
            if abs(x - r) < 1e-5:
                forces[i, 0] += 1.0
            if abs((1-x) - r) < 1e-5:
                forces[i, 0] -= 1.0
            if abs(y - r) < 1e-5:
                forces[i, 1] += 1.0
            if abs((1-y) - r) < 1e-5:
                forces[i, 1] -= 1.0
                
            # Neighbor forces
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                sum_r = radii[i] + radii[j]
                
                # If touching or very close
                if abs(dist - sum_r) < 1e-4:
                    if dist > 1e-9:
                        vec = (centers[i] - centers[j]) / dist
                        forces[i] += vec
                        forces[j] -= vec

        # Apply forces with adaptive step size
        step = 0.02 * (0.95 ** (500 - _)) # Annealing step
        centers += step * forces
        
        # Project centers back into [r, 1-r] region
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)

    # Final Radius Calculation
    radii = solve_radii(centers)
    # Clean up tiny radii that might result from numerical noise
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Verification of the constraints
import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any(): return False
    if np.isnan(radii).any(): return False

    for i in range(n):
        if radii[i] < 0 or np.isnan(radii[i]):
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

# Execute
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    print(f"Valid: {validate_packing(c, r)}")
