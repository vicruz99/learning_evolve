# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=24f7c579 sum of radii=2.606932 correctness=1.0
# stdout(first 200): Starting optimization with 15 runs... Circle 0 at (0.11588937348965045, 0.32484031136388775) with radius 0.36567572099672657 is outside the unit square Run 0: Optimization succeeded but validation fai
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # Helper function to construct initial variables
    # Order: x0, y0, r0, x1, y1, r1, ...
    def get_initial_vars(strategy='random'):
        centers = np.random.rand(n_circles, 2)
        # Start with small radii to ensure feasibility
        radii = np.full(n_circles, 0.05)
        
        # Perturb centers to avoid exact overlaps initially
        centers += np.random.randn(n_circles, 2) * 0.1
        centers = np.clip(centers, 0.1, 0.9)
        
        vars = []
        for i in range(n_circles):
            vars.append(centers[i, 0])
            vars.append(centers[i, 1])
            vars.append(radii[i])
        return np.array(vars)

    # Objective function: Minimize negative sum of radii
    def objective(vars):
        # Radii are at indices 2, 5, 8, ... (3*i + 2)
        radii = vars[2::3]
        return -np.sum(radii)

    # Gradient of objective (optional, but helpful)
    def objective_grad(vars):
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        return grad

    # Constraints
    # 1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # Linear constraints for SLSQP: c(x) >= 0. 
    # We implement as 'ineq' constraints where function returns value >= 0.
    
    def boundary_constraints(vars):
        cons = []
        for i in range(n_circles):
            idx = 3 * i
            x = vars[idx]
            y = vars[idx+1]
            r = vars[idx+2]
            # x - r >= 0
            cons.append(x - r)
            # 1 - x - r >= 0
            cons.append(1.0 - x - r)
            # y - r >= 0
            cons.append(y - r)
            # 1 - y - r >= 0
            cons.append(1.0 - y - r)
        return cons

    def boundary_constraints_jac(vars):
        # Jacobian of boundary constraints
        # Shape: (4*n, 3*n)
        jac = np.zeros((4 * n_circles, 3 * n_circles))
        for i in range(n_circles):
            col_start = 3 * i
            row_base = 4 * i
            
            # Constraint: x - r >= 0  =>  d/dx = 1, d/dr = -1
            jac[row_base, col_start] = 1.0
            jac[row_base, col_start + 2] = -1.0
            
            # Constraint: 1 - x - r >= 0 => d/dx = -1, d/dr = -1
            jac[row_base + 1, col_start] = -1.0
            jac[row_base + 1, col_start + 2] = -1.0
            
            # Constraint: y - r >= 0 => d/dy = 1, d/dr = -1
            jac[row_base + 2, col_start + 1] = 1.0
            jac[row_base + 2, col_start + 2] = -1.0
            
            # Constraint: 1 - y - r >= 0 => d/dy = -1, d/dr = -1
            jac[row_base + 3, col_start + 1] = -1.0
            jac[row_base + 3, col_start + 2] = -1.0
            
        return jac

    # 2. Non-overlap constraints: dist(i, j) >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # We will define the constraints list dynamically or use a function
    # For SLSQP, we can pass a list of constraint dictionaries.
    
    def create_overlap_constraints():
        constraints = []
        n_pairs = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                n_pairs += 1
                # Indices
                xi, yi, ri = 3*i, 3*i+1, 3*i+2
                xj, yj, rj = 3*j, 3*j+1, 3*j+2
                
                # Constraint function
                def fun(vars, i=i, j=j, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                    dx = vars[xi] - vars[xj]
                    dy = vars[yi] - vars[yj]
                    dr = vars[ri] + vars[rj]
                    return dx*dx + dy*dy - dr*dr
                
                # Jacobian
                def jac(vars, i=i, j=j, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                    grad = np.zeros(3 * n_circles)
                    dx = vars[xi] - vars[xj]
                    dy = vars[yi] - vars[yj]
                    dr = vars[ri] + vars[rj]
                    
                    # d/dx_i: 2*dx
                    grad[xi] = 2 * dx
                    # d/dy_i: 2*dy
                    grad[yi] = 2 * dy
                    # d/dr_i: -2*dr
                    grad[ri] = -2 * dr
                    
                    # d/dx_j: -2*dx
                    grad[xj] = -2 * dx
                    # d/dy_j: -2*dy
                    grad[yj] = -2 * dy
                    # d/dr_j: -2*dr
                    grad[rj] = -2 * dr
                    
                    return grad

                constraints.append({
                    'type': 'ineq',
                    'fun': fun,
                    'jac': jac
                })
        return constraints

    overlap_constraints = create_overlap_constraints()
    
    # Combine constraints
    # We need to return a list of constraint dicts. 
    # The boundary constraints need to be wrapped similarly or handled.
    # SLSQP accepts a list of dicts.
    
    # Wrapper for boundary constraints to return a single vector for all
    # But SLSQP handles list of constraints or one dict with vector fun.
    # Let's use list of dicts for flexibility, or one big dict.
    # One big dict for boundary constraints might be faster.
    
    boundary_constraint_dict = {
        'type': 'ineq',
        'fun': boundary_constraints,
        'jac': boundary_constraints_jac
    }
    
    all_constraints = [boundary_constraint_dict] + overlap_constraints
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Multi-start optimization
    n_runs = 15
    print(f"Starting optimization with {n_runs} runs...")
    
    for run in range(n_runs):
        # Initial guess: Random distribution
        # Sometimes a grid is better, but random is robust for non-convex
        x0 = get_initial_vars()
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=all_constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if result.success or result.nit > 0:
                current_sum = -result.fun
                if current_sum > best_sum:
                    # Extract results
                    vars_opt = result.x
                    centers = np.zeros((n_circles, 2))
                    radii = np.zeros(n_circles)
                    for i in range(n_circles):
                        centers[i, 0] = vars_opt[3*i]
                        centers[i, 1] = vars_opt[3*i+1]
                        radii[i] = vars_opt[3*i+2]
                    
                    # Validate before accepting
                    if validate_packing(centers, radii):
                        best_sum = current_sum
                        best_centers = centers.copy()
                        best_radii = radii.copy()
                        print(f"Run {run}: Valid packing found. Sum radii = {best_sum:.6f}")
                    else:
                        print(f"Run {run}: Optimization succeeded but validation failed.")
                else:
                     # Even if validation fails, keep track of best objective found so far?
                     # No, we need valid packing.
                     pass
        except Exception as e:
            print(f"Run {run} failed with error: {e}")

    if best_centers is None:
        # Fallback: Simple grid if optimization fails completely
        # 5x5 grid + 1 small circle?
        # Just return a valid simple solution
        centers = np.array([(0.1, 0.1) for _ in range(26)]) # Invalid overlap
        # Better fallback: Small circles in grid
        coords = np.random.rand(n_circles, 2) * 0.8 + 0.1
        radii = np.full(n_circles, 0.01)
        best_centers = coords
        best_radii = radii
        best_sum = 0.26

    return best_centers, best_radii, best_sum
