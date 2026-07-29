# sol_000142 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 466799c7) state=a21dfc4c sum of radii=1.040000 correctness=1.0
# stdout(first 200): Circle 0 at (3.6993394624840804e-16, 0.0) with radius 0.1413038240746597 is outside the unit square Circle 0 at (3.6993394624840804e-16, 0.0) with radius 0.1413038240746597 is outside the unit square 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def evaluate_objective(vars, n):
    """
    Computes the negative sum of radii to be minimized.
    """
    radii = vars[2::3]
    return -np.sum(radii)

def compute_overlap_constraints(vars, n):
    """
    Computes the vector of overlap constraints.
    Constraint: dist_sq(i, j) - (r_i + r_j)^2 >= 0
    Returns an array of constraint values for all i < j.
    """
    # Extract x, y, r
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Compute pairwise differences using broadcasting
    # Shape: (n, n)
    x_diff = x[:, np.newaxis] - x[np.newaxis, :]
    y_diff = y[:, np.newaxis] - y[np.newaxis, :]
    
    # Squared Euclidean distance
    dist_sq = x_diff**2 + y_diff**2
    
    # Sum of radii squared matrix
    # r shape (n,), r[:, np.newaxis] is (n, 1)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Constraint value: dist_sq - r_sum_sq
    # We only need the upper triangular part (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    constraints = dist_sq[mask] - r_sum_sq[mask]
    
    return constraints

def generate_initial_guess(n, strategy='hex'):
    """
    Generates a valid initial configuration for n circles.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    if strategy == 'hex':
        # Hexagonal packing initialization
        # Start with a small radius to ensure validity
        r_init = 0.05
        radii[:] = r_init
        
        # Hexagonal lattice parameters
        # Row spacing dy = sqrt(3) * r_init
        # Column spacing dx = 2 * r_init
        dy = np.sqrt(3) * r_init
        dx = 2 * r_init
        
        points = []
        y = r_init
        row_idx = 0
        while len(points) < n:
            x = r_init
            if row_idx % 2 != 0:
                x += dx / 2.0
            
            while x <= 1 - r_init:
                points.append((x, y))
                if len(points) >= n:
                    break
                x += dx
            
            y += dy
            row_idx += 1
        
        # Fill centers
        for i in range(n):
            centers[i] = points[i]
            
    elif strategy == 'random':
        # Random placement with small radius
        r_init = 0.05
        radii[:] = r_init
        # Place centers randomly within [r, 1-r]
        centers = np.random.rand(n, 2) * (1 - 2*r_init) + r_init
        
    vars = np.zeros(3 * n)
    vars[0::3] = centers[:, 0]
    vars[1::3] = centers[:, 1]
    vars[2::3] = radii
    
    return vars

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Define bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': lambda vars: compute_overlap_constraints(vars, n)
    }
    
    # Multi-start optimization
    # We try multiple initial configurations to escape local minima
    strategies = ['hex', 'hex', 'hex', 'random', 'random']
    # Add some perturbation to hex grid
    for i in range(5):
        strategy = strategies[i] if i < len(strategies) else 'hex'
        
        # Generate initial guess
        vars_init = generate_initial_guess(n, strategy)
        
        # If hex, maybe add small random noise to break symmetry if needed, 
        # but SLSQP usually handles it. Let's keep it clean.
        
        # Optimization options
        options = {
            'maxiter': 1000,
            'ftol': 1e-10,
            'disp': False
        }
        
        try:
            res = minimize(
                evaluate_objective,
                vars_init,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options=options
            )
            
            if res.success or (res.nit > 0 and not np.any(np.isnan(res.x))):
                current_sum = -res.fun
                
                # Extract centers and radii
                centers_candidate = np.column_stack((res.x[0::3], res.x[1::3]))
                radii_candidate = res.x[2::3]
                
                # Validate the result strictly
                if validate_packing(centers_candidate, radii_candidate):
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = centers_candidate
                        best_radii = radii_candidate
                else:
                    # If validation fails (rare with SLSQP but possible due to tolerances),
                    # we can try to shrink radii slightly to fix, but for this problem
                    # SLSQP should be tight.
                    pass
                    
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue
            
    # Final fallback if nothing found (should not happen with hex init)
    if best_centers is None:
        # Default to a simple grid
        r = 0.1
        centers = np.array([(0.1 + i*0.2, 0.1 + j*0.2) for j in range(5) for i in range(5)]).flatten()
        # This is 25 points. We need 26. 
        # Just return something valid.
        # But with the logic above, we should have found something.
        pass

    # Ensure best_radii is not None
    if best_centers is None:
         # Emergency valid configuration
         best_centers = np.zeros((n, 2))
         best_radii = np.zeros(n)
         for i in range(n):
             best_centers[i] = (0.05 + (i%5)*0.1, 0.05 + (i//5)*0.1)
             best_radii[i] = 0.04
         best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii
