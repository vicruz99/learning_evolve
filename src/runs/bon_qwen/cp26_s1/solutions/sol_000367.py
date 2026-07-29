# sol_000367 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b037cf31) state=64234a04 sum of radii=2.400000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a hybrid optimization approach starting from a 5x5 grid plus a central circle.
    """
    n = 26
    
    # Initialize centers: 5x5 grid for first 25, 1 in center
    centers = np.zeros((n, 2))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            # Start with a tight grid (r=0.1 implies step 0.2)
            # Center at 0.1 + 0.2*k
            centers[idx] = [0.1 + 0.2 * j, 0.1 + 0.2 * i]
    
    # Center circle at 0.5, 0.5
    centers[25] = [0.5, 0.5]
    
    # Initialize radii: equal small radii to start valid
    radii = np.ones(n) * 0.05

    # Objective: Minimize negative sum of radii
    def objective(vars_):
        r_vals = vars_
        return -np.sum(r_vals)

    # Constraints:
    # 1. Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    # 2. Non-overlap: dist(i,j) >= r_i + r_j

    def get_constraints():
        cons = []
        
        # Boundary constraints for each circle
        for i in range(n):
            # We need to extract centers and radii from the optimization variables
            # But here we assume centers are fixed and only radii are optimized?
            # To be more flexible, let's optimize radii only, keeping centers fixed?
            # No, moving centers might help, but keeping centers fixed simplifies the problem significantly
            # and is a strong heuristic for grid-based packings.
            pass

        # Let's just optimize radii for the fixed centers first.
        # If we want to optimize centers, the variable space is 52 (26*2) + 26 = 78 dims.
        # That might be slow or get stuck.
        # Let's try optimizing radii first with fixed centers.
        return []

    # Optimization of radii only
    # Variables: 26 radii
    def radii_objective(r):
        return -np.sum(r)

    def radii_constraints():
        cons = []
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            # r <= x, r <= 1-x, r <= y, r <= 1-y
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: r[i] - x, 'args': ()}) # This lambda is bad
            # Wait, I cannot use closures/lambda with args easily in this structure if I strictly follow "no closures from function nesting" 
            # but I can define helper functions.
            # Actually, standard scipy constraints take args.
            pass
        
        # Non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                cons.append({'type': 'ineq', 'fun': lambda r, i=i, j=j, d=dist: d - (r[i] + r[j])})
                
        return cons

    # Better constraint definition without closure issues
    def bound_constraint_x(r, i):
        return centers[i, 0] - r[i]
    def bound_constraint_x1(r, i):
        return 1.0 - centers[i, 0] - r[i]
    def bound_constraint_y(r, i):
        return centers[i, 1] - r[i]
    def bound_constraint_y1(r, i):
        return 1.0 - centers[i, 1] - r[i]

    def overlap_constraint(r, i, j):
        dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
        return dist - (r[i] + r[j])

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': bound_constraint_x, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_constraint_x1, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_constraint_y, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_constraint_y1, 'args': (i,)})
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j)})

    # Bounds for radii: [0, 0.5]
    bounds = [(0, 0.5) for _ in range(n)]

    # Run optimization
    # Use SLSQP for bound and constraint handling
    res = minimize(radii_objective, radii, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})

    # If optimization fails to improve significantly or gets stuck, we might need to perturb centers
    # But for a 5x5 grid + center, optimizing radii is usually sufficient to find the local optimum
    # where the grid circles touch each other and the center circle.
    
    final_radii = res.x
    
    # Validate and return
    # Re-check validity just in case
    if validate_packing(centers, final_radii):
        return centers, final_radii, np.sum(final_radii)
    else:
        # Fallback to a known valid packing if optimization breaks
        # Return the initial valid state
        return centers, np.ones(n) * 0.05, 26 * 0.05

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True
