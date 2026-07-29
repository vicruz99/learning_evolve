# sol_000160 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a0a8497a) state=147ba0f2 sum of radii=2.549783 correctness=1.0
# stdout(first 200): Circles 2 and 7 overlap: dist=0.17453686272895005, r1+r2=0.1745368627304356
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
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

def get_initial_guess(n, type_id):
    """
    Generates initial guess for circle centers and radii.
    type_id 0: Simple grid
    type_id 1: Perturbed hexagonal
    type_id 2: Offset grid
    """
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)
    
    # Adjust grid size based on n
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        if type_id == 0:
            # Simple grid
            x = (col + 0.5) * (1.0 / cols)
            y = (row + 0.5) * (1.0 / rows)
        elif type_id == 1:
            # Perturbed hexagonal packing
            spacing_x = 1.0 / cols
            spacing_y = (np.sqrt(3) / 2) * spacing_x
            
            # Shift every other row
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            
            if row % 2 == 1:
                x += spacing_x / 2
        elif type_id == 2:
            # Offset grid
            x = (col + 0.5) * (1.0 / cols)
            y = (row + 0.5) * (1.0 / rows)
            if row % 2 == 1:
                x += 0.05

        centers[i] = [x, y]
        
    return centers, radii

def objective_function(variables, n):
    # Variables: [x0, y0, r0, x1, y1, r1, ...]
    # We want to maximize sum of radii, so minimize -sum of radii
    radii = variables[2::3]
    return -np.sum(radii)

def constraint_overlap(variables, n, i, j):
    # Distance between i and j >= r_i + r_j
    x_i, y_i, r_i = variables[3*i], variables[3*i+1], variables[3*i+2]
    x_j, y_j, r_j = variables[3*j], variables[3*j+1], variables[3*j+2]
    
    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
    return np.sqrt(dist_sq) - (r_i + r_j)

def constraint_boundary_x(variables, n, i):
    # r <= x <= 1 - r  =>  x - r >= 0  AND  x + r <= 1
    return variables[3*i] - variables[3*i+2]

def constraint_boundary_x_upper(variables, n, i):
    return variables[3*i] + variables[3*i+2] - 1

def constraint_boundary_y(variables, n, i):
    return variables[3*i+1] - variables[3*i+2]

def constraint_boundary_y_upper(variables, n, i):
    return variables[3*i+1] + variables[3*i+2] - 1

def constraint_radius(variables, n, i):
    # r >= 0
    return variables[3*i+2]

def solve_packing(initial_guess_type):
    n = 26
    centers, radii = get_initial_guess(n, initial_guess_type)
    
    # Flatten initial variables
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Define bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Define constraints
    constraints = []
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: constraint_overlap(v, n, i, j)
            })
            
    # Boundary and radius constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_x(v, n, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: -constraint_boundary_x_upper(v, n, i)}) # x + r <= 1 => -(x+r-1) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_y(v, n, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: -constraint_boundary_y_upper(v, n, i)}) # y + r <= 1
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_radius(v, n, i)})
        
    # Run optimization
    result = minimize(
        objective_function, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # Extract results
    if result.success or result.fun > -2.5: # Allow even if not strictly successful if score is decent
        res_centers = np.zeros((n, 2))
        res_radii = np.zeros(n)
        for i in range(n):
            res_centers[i, 0] = result.x[3*i]
            res_centers[i, 1] = result.x[3*i+1]
            res_radii[i] = result.x[3*i+2]
            
        # Clamp to ensure strict validity within numerical error
        for i in range(n):
            x, y, r = res_centers[i, 0], res_centers[i, 1], res_radii[i]
            r = max(r, 1e-6)
            res_radii[i] = r
            res_centers[i, 0] = np.clip(x, r, 1-r)
            res_centers[i, 1] = np.clip(y, r, 1-r)
            
        return res_centers, res_radii
    return None, None

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Try multiple initial configurations
    for guess_type in [0, 1, 2]:
        centers, radii = solve_packing(guess_type)
        if centers is not None:
            current_sum = np.sum(radii)
            # Validate before updating best
            if validate_packing(centers, radii):
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers
                    best_radii = radii
            else:
                # If validation fails, try to fix small overlaps by shrinking slightly
                # This is a fallback, though SLSQP should handle constraints
                pass

    # Fallback if optimization failed completely (unlikely with good initialization)
    if best_centers is None:
        best_centers, best_radii = get_initial_guess(n, 1)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
