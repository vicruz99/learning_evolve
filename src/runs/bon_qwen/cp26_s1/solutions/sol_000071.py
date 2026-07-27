# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfef56bb) state=91bc70ef sum of radii=2.607014 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def check_constraints(params, n=26):
    """
    Helper function to compute constraint violations.
    Returns a flat array of constraint values.
    We want these to be <= 0.
    """
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    violations = []
    
    # Boundary constraints: x - r >= 0  =>  r - x <= 0
    # x + r <= 1  =>  x + r - 1 <= 0
    # y - r >= 0  =>  r - y <= 0
    # y + r <= 1  =>  y + r - 1 <= 0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        violations.append(r - x)
        violations.append(x + r - 1)
        violations.append(r - y)
        violations.append(y + r - 1)
        
    # Overlap constraints: dist >= r1 + r2  =>  (r1 + r2)^2 - dist^2 <= 0
    # Using squared distance to avoid sqrt and keep it smooth
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            violations.append(r_sum * r_sum - dist_sq)
            
    return np.array(violations)

def get_constraints(n=26):
    """
    Returns a list of constraint dictionaries for scipy.
    """
    cons = []
    # We treat all constraints as inequalities: fun(x) <= 0
    # We can pass a single function that returns a vector, but scipy 
    # minimize with SLSQP accepts a list of dicts with 'type': 'ineq' 
    # where fun(x) >= 0. So we negate our violations.
    
    def con_func(params):
        # violations >= 0 means valid? 
        # Our check_constraints returns values that should be <= 0.
        # So we return -violations to make them >= 0.
        return -check_constraints(params, n)

    cons.append({'type': 'ineq', 'fun': con_func})
    return cons

def get_bounds(n=26):
    """
    Bounds for variables: centers in [0, 1], radii >= 0.
    """
    # centers: 2*n variables
    bounds_centers = [(0, 1) for _ in range(2*n)]
    # radii: n variables
    bounds_radii = [(0, 1) for _ in range(n)] # Upper bound 1 is loose but safe
    return bounds_centers + bounds_radii

def generate_initial_guess(n=26):
    """
    Generate a hexagonal packing initial guess.
    """
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Start with small radius
    
    # Hexagonal packing logic
    # Row height = sqrt(3)/2 * diameter = sqrt(3) * r
    # But since r is variable, let's just place centers in a grid pattern first
    # and let optimizer scale radii.
    
    # A simple grid initialization
    # sqrt(26) approx 5.1
    cols = 6
    rows = 5
    step_x = 1.0 / (cols + 1)
    step_y = 1.0 / (rows + 1)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                x = (c + 1) * step_x
                y = (r + 1) * step_y
                # Shift alternate rows for hexagonal-ish start
                if r % 2 == 1:
                    x += step_x / 2
                centers[idx] = [x, y]
                idx += 1
    
    # Fill remaining if any (though 5x6=30 > 26)
    # Adjust indices
    return centers, radii

def run_packing():
    n = 26
    
    best_sum_radii = 0
    best_centers = None
    best_radii = None
    
    # We will run optimization several times with different initial guesses
    num_attempts = 5
    
    for attempt in range(num_attempts):
        # Generate initial guess
        centers, radii = generate_initial_guess(n)
        
        # Add some random noise to escape local minima
        if attempt > 0:
            noise_scale = 0.05
            centers += np.random.uniform(-noise_scale, noise_scale, centers.shape)
            radii += np.random.uniform(-noise_scale, noise_scale, radii.shape)
            # Clamp
            centers = np.clip(centers, 0.01, 0.99)
            radii = np.clip(radii, 0.01, 0.5)
            
        # Flatten parameters
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Define objective: maximize sum(radii) => minimize -sum(radii)
        def objective(params):
            r = params[2*n:]
            return -np.sum(r)
        
        # Constraints
        cons = get_constraints(n)
        
        # Bounds
        bounds = get_bounds(n)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                centers_opt = res.x[:2*n].reshape(n, 2)
                radii_opt = res.x[2*n:]
                
                # Validate manually to be sure (checking for overlaps with tolerance)
                valid = True
                for i in range(n):
                    x, y = centers_opt[i]
                    r = radii_opt[i]
                    if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
                        valid = False
                        break
                    for j in range(i + 1, n):
                        dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                        if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    current_sum = np.sum(radii_opt)
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = centers_opt
                        best_radii = radii_opt
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue
            
    # If optimization didn't find a valid solution or better one, 
    # return a fallback or the best found.
    # As a fallback, ensure we return something valid.
    if best_centers is None:
        # Fallback to a simple grid with small radius
        centers, radii = generate_initial_guess(n)
        # Scale down to be safe
        radii[:] = 0.04
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii
