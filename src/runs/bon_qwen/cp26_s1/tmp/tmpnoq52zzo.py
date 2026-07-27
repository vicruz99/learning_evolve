import numpy as np
from scipy.optimize import minimize
import math

N_CIRCLES = 26

def get_hexagonal_initialization(n):
    """
    Generates an initial guess for centers and radii based on a hexagonal grid.
    """
    # Estimate radius for n circles in unit square using hexagonal density approximation
    # Area ~ n * pi * r^2 / (pi / sqrt(12)) = n * 2 * sqrt(3) * r^2 <= 1
    # r^2 <= 1 / (2 * sqrt(3) * n)
    r_est = math.sqrt(1.0 / (2.0 * math.sqrt(3.0) * n)) * 0.95 # 0.95 safety factor
    
    centers = []
    row = 0
    # Hexagonal packing: rows shifted
    while len(centers) < n:
        # y coordinate: start at r, step sqrt(3)*r
        y = r_est + row * math.sqrt(3.0) * r_est
        
        # If row is too high, break (though with r_est it should fit)
        if y + r_est > 1.0 + 1e-9:
            # If we run out of vertical space, try reducing r_est slightly? 
            # Or just stop. But with n=26, 6 rows fit easily.
            # Just ensure we don't generate invalid points.
            pass
            
        # x coordinates: step 2*r. Shift by r for odd rows.
        shift = r_est if (row % 2 == 1) else 0.0
        col = 0
        while True:
            x = r_est + col * 2.0 * r_est + shift
            
            if x + r_est > 1.0 + 1e-9:
                break
            
            if y + r_est <= 1.0 + 1e-9:
                centers.append((x, y))
                if len(centers) >= n:
                    break
            col += 1
        
        row += 1
        if row > 50: # Safety break
            break
            
    # Fill remaining if any (should not happen with good r_est)
    while len(centers) < n:
        centers.append((0.5, 0.5))
        
    centers = np.array(centers[:n])
    radii = np.full(n, r_est)
    
    # Construct variable vector v = [x0, y0, r0, x1, y1, r1, ...]
    v = np.zeros(3 * n)
    for i in range(n):
        v[3*i] = centers[i, 0]
        v[3*i+1] = centers[i, 1]
        v[3*i+2] = radii[i]
        
    return v

def objective(v):
    """
    Objective function: minimize negative sum of radii.
    Variables order: x0, y0, r0, x1, y1, r1, ...
    """
    radii = v[2::3]
    return -np.sum(radii)

def objective_grad(v):
    """
    Gradient of the objective function.
    """
    grad = np.zeros_like(v)
    # d(-sum(r))/dr_i = -1
    grad[2::3] = -1.0
    return grad

def constraint_values(v):
    """
    Computes constraint values.
    Returns array of values that must be >= 0.
    Constraints:
    1. Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    2. Non-overlap: dist^2 >= (r_i + r_j)^2
    """
    c = []
    
    # Boundary constraints
    for i in range(N_CIRCLES):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x - r >= 0
        c.append(v[idx_x] - v[idx_r])
        # 1 - x - r >= 0
        c.append(1.0 - v[idx_x] - v[idx_r])
        # y - r >= 0
        c.append(v[idx_y] - v[idx_r])
        # 1 - y - r >= 0
        c.append(1.0 - v[idx_y] - v[idx_r])
        
    # Pairwise non-overlap constraints
    # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    for i in range(N_CIRCLES):
        idx_xi = 3 * i
        idx_yi = 3 * i + 1
        idx_ri = 3 * i + 2
        
        xi = v[idx_xi]
        yi = v[idx_yi]
        ri = v[idx_ri]
        
        for j in range(i + 1, N_CIRCLES):
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            xj = v[idx_xj]
            yj = v[idx_yj]
            rj = v[idx_rj]
            
            dx = xi - xj
            dy = yi - yj
            dr_sum = ri + rj
            
            val = dx*dx + dy*dy - dr_sum*dr_sum
            c.append(val)
            
    return np.array(c)

def constraint_jacobian(v):
    """
    Computes the Jacobian matrix of the constraints.
    Rows correspond to constraints, columns to variables.
    """
    n_vars = 3 * N_CIRCLES
    # Number of boundary constraints: 4 * N
    # Number of pairwise constraints: N*(N-1)/2
    n_constraints = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    
    J = np.zeros((n_constraints, n_vars))
    row = 0
    
    # Boundary constraints Jacobian
    for i in range(N_CIRCLES):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # Constraint: x - r >= 0
        # d/dx = 1, d/dr = -1
        J[row, idx_x] = 1.0
        J[row, idx_r] = -1.0
        row += 1
        
        # Constraint: 1 - x - r >= 0
        # d/dx = -1, d/dr = -1
        J[row, idx_x] = -1.0
        J[row, idx_r] = -1.0
        row += 1
        
        # Constraint: y - r >= 0
        # d/dy = 1, d/dr = -1
        J[row, idx_y] = 1.0
        J[row, idx_r] = -1.0
        row += 1
        
        # Constraint: 1 - y - r >= 0
        # d/dy = -1, d/dr = -1
        J[row, idx_y] = -1.0
        J[row, idx_r] = -1.0
        row += 1
        
    # Pairwise constraints Jacobian
    # Constraint: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(N_CIRCLES):
        idx_xi = 3 * i
        idx_yi = 3 * i + 1
        idx_ri = 3 * i + 2
        
        xi = v[idx_xi]
        yi = v[idx_yi]
        ri = v[idx_ri]
        
        for j in range(i + 1, N_CIRCLES):
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            xj = v[idx_xj]
            yj = v[idx_yj]
            rj = v[idx_rj]
            
            dx = xi - xj
            dy = yi - yj
            dr_sum = ri + rj
            
            # Gradient wrt xi: 2*dx
            J[row, idx_xi] = 2.0 * dx
            # Gradient wrt xj: -2*dx
            J[row, idx_xj] = -2.0 * dx
            # Gradient wrt yi: 2*dy
            J[row, idx_yi] = 2.0 * dy
            # Gradient wrt yj: -2*dy
            J[row, idx_yj] = -2.0 * dy
            # Gradient wrt ri: -2*(ri+rj)
            J[row, idx_ri] = -2.0 * dr_sum
            # Gradient wrt rj: -2*(ri+rj)
            J[row, idx_rj] = -2.0 * dr_sum
            
            row += 1
            
    return J

def run_packing():
    # Define constraints dictionary for scipy
    # SLSQP expects constraints in format: {'type': 'ineq', 'fun': func, 'jac': func}
    # However, passing a single function for all inequalities is efficient.
    
    cons = (
        {
            'type': 'ineq',
            'fun': constraint_values,
            'jac': constraint_jacobian
        }
    )
    
    best_v = None
    best_val = -np.inf
    
    # Run optimization multiple times with perturbed initializations
    # to escape local minima.
    for attempt in range(5):
        # Get hexagonal initialization
        v0 = get_hexagonal_initialization(N_CIRCLES)
        
        # Add small random noise
        noise = np.random.normal(0, 0.01, size=v0.shape)
        # Ensure radii don't become negative due to noise
        noise[2::3] = np.abs(noise[2::3]) 
        # Ensure boundaries are respected roughly (clip noise if needed, but small is ok)
        
        v_start = v0 + noise
        
        # Run optimizer
        res = minimize(
            objective,
            v_start,
            method='SLSQP',
            jac=objective_grad,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success or (res.fun < -best_val): # We want to maximize sum, so minimize -sum. Lower fun is better.
            if res.fun < best_val: # Since fun is negative sum, smaller is larger sum
                best_val = res.fun
                best_v = res.x
    
    if best_v is None:
        # Fallback to single run if all failed (unlikely)
        v0 = get_hexagonal_initialization(N_CIRCLES)
        res = minimize(objective, v0, method='SLSQP', jac=objective_grad, constraints=cons)
        best_v = res.x
        
    # Extract results
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    
    for i in range(N_CIRCLES):
        centers[i, 0] = best_v[3*i]
        centers[i, 1] = best_v[3*i+1]
        radii[i] = best_v[3*i+2]
        
    sum_radii = np.sum(radii)
    
    # Final safety clamp (numerical errors might cause tiny violations)
    # Though optimizer should satisfy them.
    # Just in case, enforce non-negative radii
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(sum_radii)