# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82191eeb) state=4180566a sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def compute_dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

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

def get_constraints(n):
    """
    Returns a list of constraint dictionaries for scipy.optimize.minimize (SLSQP).
    Constraints are of the form c(x) >= 0.
    """
    constraints = []
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # Overlap constraints:
    # dist(i, j) - (r_i + r_j) >= 0
    
    # To improve performance, we can define functions that compute all boundary constraints
    # and all overlap constraints in vectorized ways if possible, but SLSQP expects
    # functions returning arrays.
    
    # Let's define a function for boundary constraints
    def boundary_constraints(vars):
        # vars is 1D array: [x0, y0, r0, x1, y1, r1, ...]
        # Reshape to (n, 3) for easier indexing
        data = vars.reshape(n, 3)
        x = data[:, 0]
        y = data[:, 1]
        r = data[:, 2]
        
        # Constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        c1 = x - r
        c2 = 1 - x - r
        c3 = y - r
        c4 = 1 - y - r
        
        return np.concatenate([c1, c2, c3, c4])

    constraints.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Overlap constraints
    # This is O(N^2). For N=26, 325 constraints.
    # We define a function that computes all of them.
    def overlap_constraints(vars):
        data = vars.reshape(n, 3)
        x = data[:, 0]
        y = data[:, 1]
        r = data[:, 2]
        
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                overlaps.append(dist - (r[i] + r[j]))
        
        return np.array(overlaps)

    constraints.append({'type': 'ineq', 'fun': overlap_constraints})
    
    return constraints

def solve_from_initial(initial_vars, n):
    """
    Runs optimization starting from initial_vars.
    Returns (centers, radii, sum_radii).
    """
    bounds = []
    for i in range(n):
        # x in [0, 1]
        bounds.append((0.0, 1.0))
        # y in [0, 1]
        bounds.append((0.0, 1.0))
        # r in [0, 0.5] (radius cannot be more than 0.5)
        bounds.append((0.0, 0.5))
        
    constraints = get_constraints(n)
    
    # Objective: minimize -sum(r)
    def objective(vars):
        data = vars.reshape(n, 3)
        r = data[:, 2]
        return -np.sum(r)
        
    try:
        # Use SLSQP
        res = scipy.optimize.minimize(
            objective, 
            initial_vars, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if res.success or res.nit > 100: # Accept if converged or ran for a while
            data = res.x.reshape(n, 3)
            centers = data[:, :2]
            radii = data[:, 2]
            sum_r = np.sum(radii)
            return centers, radii, sum_r
        else:
            return None, None, -1.0
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None, -1.0

def generate_random_init(n):
    """Generate random valid initial positions with small radii."""
    # Place centers randomly in [0.1, 0.9] to be safe from boundaries
    centers = np.random.uniform(0.1, 0.9, size=(n, 2))
    # Small radii
    radii = np.full(n, 0.01)
    
    # Flatten to [x0, y0, r0, x1, y1, r1, ...]
    vars = np.zeros(3 * n)
    for i in range(n):
        vars[3*i] = centers[i, 0]
        vars[3*i+1] = centers[i, 1]
        vars[3*i+2] = radii[i]
    return vars

def generate_grid_init(n):
    """Generate a grid-based initialization.
    Try to fit 25 circles in a 5x5 grid and one extra in a hole."""
    vars = np.zeros(3 * n)
    
    # 5x5 grid for first 25
    # Grid spacing 0.2, starting at 0.1. Radius 0.1.
    # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
    count = 0
    for r_idx in range(5):
        for c_idx in range(5):
            if count < 25:
                x = 0.1 + 0.2 * c_idx
                y = 0.1 + 0.2 * r_idx
                r = 0.1
                vars[3*count] = x
                vars[3*count+1] = y
                vars[3*count+2] = r
                count += 1
    
    # 26th circle
    # Place in a hole. Hole at (0.2, 0.2) is surrounded by (0.1,0.1), (0.3,0.1), etc.
    # Distance is sqrt(0.02) approx 0.1414.
    # Radius can be 0.1414 - 0.1 = 0.0414.
    if n > 25:
        vars[3*25] = 0.2
        vars[3*25+1] = 0.2
        vars[3*25+2] = 0.04
        
    return vars

def generate_hex_init(n):
    """Attempt a hexagonal packing initialization."""
    vars = np.zeros(3 * n)
    count = 0
    
    # Try to fit rows.
    # Approximate radius 0.09 to fit width.
    # Hex spacing: dx = 2r, dy = r*sqrt(3).
    # Row shift dx/2 = r.
    r_est = 0.085 
    dx = 2 * r_est
    dy = r_est * math.sqrt(3)
    shift = r_est
    
    # We need to fill the square.
    # Let's try to generate points and select best n?
    # Or just fill rows.
    
    y = r_est
    row = 0
    while count < n:
        x = r_est
        if row % 2 == 1:
            x = r_est + shift
        
        while x + r_est <= 1.0 + 1e-9 and count < n:
            vars[3*count] = x
            vars[3*count+1] = y
            vars[3*count+2] = r_est
            count += 1
            x += dx
        
        y += dy
        row += 1
        
    return vars

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple strategies
    strategies = [
        generate_grid_init,
        generate_random_init,
        generate_random_init,
        generate_hex_init
    ]
    
    for strategy in strategies:
        try:
            if strategy == generate_random_init:
                np.random.seed(None) # Random seed
            
            init_vars = strategy(n)
            
            # Run optimization
            centers, radii, s = solve_from_initial(init_vars, n)
            
            if centers is not None:
                # Validate
                if validate_packing(centers, radii):
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers.copy()
                        best_radii = radii.copy()
        except Exception as e:
            print(f"Strategy failed: {e}")
            continue

    # Fallback if nothing worked (should not happen with grid init)
    if best_centers is None:
        # Return a simple valid packing (small circles)
        centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        radii = np.full(n, 0.01)
        best_sum = 0.26
        
    # Final check
    if validate_packing(best_centers, best_radii):
        return best_centers, best_radii, best_sum
    else:
        # If validation fails (shouldn't), return a safe fallback
        print("Final validation failed, returning safe fallback.")
        centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        radii = np.full(n, 0.005)
        return centers, radii, np.sum(radii)

# Note: The validation function is provided in the prompt, but for the code to be self-contained 
# and runnable in the environment, we include a copy or assume it's available. 
# The prompt says "We will run the below validation function", implying it's external. 
# However, to be safe and follow "Make all helper functions top level", I included it.
# If the environment provides it, my definition might shadow it, but it's identical.
