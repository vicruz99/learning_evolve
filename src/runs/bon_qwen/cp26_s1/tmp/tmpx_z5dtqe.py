import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix

def generate_initial_config(n, seed=0):
    """
    Generate a feasible initial configuration for n circles.
    Uses a hexagonal grid pattern with some random perturbation.
    """
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05)
    
    # Pattern for 26 circles: 5 rows
    # Counts: 5, 6, 5, 6, 4 = 26
    row_counts = [5, 6, 5, 6, 4]
    y_vals = np.linspace(0.12, 0.88, 5)
    
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = y_vals[r_idx]
        # Stagger x coordinates for hexagonal packing
        if r_idx % 2 == 0:
            x_start = 0.15
        else:
            x_start = 0.15 + 0.09
            
        if count == 0: 
            continue
            
        # Distribute x coordinates
        xs = np.linspace(x_start, 1 - x_start, count)
        
        for i in range(count):
            # Add small random perturbation to break symmetry and help optimization
            jitter = np.random.uniform(-0.02, 0.02, 2)
            centers[idx, 0] = np.clip(xs[i] + jitter[0], 0.1, 0.9)
            centers[idx, 1] = np.clip(y + jitter[1], 0.1, 0.9)
            idx += 1
            
    return centers, radii

def objective(v, n):
    """
    Objective function: minimize negative sum of radii.
    v is flattened vector [x1, y1, r1, x2, y2, r2, ...]
    """
    r = v[2::3]
    return -np.sum(r)

def objective_jac(v, n):
    """
    Jacobian of the objective function.
    """
    jac = np.zeros_like(v)
    # d/dr (-sum(r)) = -1 for all r components
    jac[2::3] = -1.0
    return jac

def constraints_fun(v, n):
    """
    Computes constraint values.
    Returns an array of constraint values (all must be >= 0).
    Order:
    1. x_i - r_i >= 0 (n constraints)
    2. 1 - x_i - r_i >= 0 (n constraints)
    3. y_i - r_i >= 0 (n constraints)
    4. 1 - y_i - r_i >= 0 (n constraints)
    5. (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0 (n*(n-1)/2 constraints)
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = np.zeros(4*n + n*(n-1)//2)
    
    # Boundary constraints
    c[0:n] = x - r
    c[n:2*n] = 1 - x - r
    c[2*n:3*n] = y - r
    c[3*n:4*n] = 1 - y - r
    
    # Non-overlap constraints
    k = 4*n
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            c[k] = dx*dx + dy*dy - dr*dr
            k += 1
            
    return c

def constraints_jac(v, n):
    """
    Computes the Jacobian of the constraints.
    Returns a sparse matrix of shape (num_constraints, 3*n).
    """
    num_constraints = 4*n + n*(n-1)//2
    num_vars = 3*n
    
    row = []
    col = []
    data = []
    
    # Boundary constraints Jacobians
    # 1. x_i - r_i >= 0
    for i in range(n):
        idx = i
        row.append(idx); col.append(3*i); data.append(1.0)       # d/dx_i
        row.append(idx); col.append(3*i+2); data.append(-1.0)    # d/dr_i
        
    # 2. 1 - x_i - r_i >= 0
    for i in range(n):
        idx = n + i
        row.append(idx); col.append(3*i); data.append(-1.0)      # d/dx_i
        row.append(idx); col.append(3*i+2); data.append(-1.0)    # d/dr_i
        
    # 3. y_i - r_i >= 0
    for i in range(n):
        idx = 2*n + i
        row.append(idx); col.append(3*i+1); data.append(1.0)     # d/dy_i
        row.append(idx); col.append(3*i+2); data.append(-1.0)    # d/dr_i
        
    # 4. 1 - y_i - r_i >= 0
    for i in range(n):
        idx = 3*n + i
        row.append(idx); col.append(3*i+1); data.append(-1.0)    # d/dy_i
        row.append(idx); col.append(3*i+2); data.append(-1.0)    # d/dr_i
        
    # Non-overlap constraints Jacobians
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    k = 4*n
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            
            # Derivatives
            # d/dx_i: 2*dx
            row.append(k); col.append(3*i); data.append(2*dx)
            # d/dx_j: -2*dx
            row.append(k); col.append(3*j); data.append(-2*dx)
            # d/dy_i: 2*dy
            row.append(k); col.append(3*i+1); data.append(2*dy)
            # d/dy_j: -2*dy
            row.append(k); col.append(3*j+1); data.append(-2*dy)
            # d/dr_i: -2*dr
            row.append(k); col.append(3*i+2); data.append(-2*dr)
            # d/dr_j: -2*dr
            row.append(k); col.append(3*j+2); data.append(-2*dr)
            
            k += 1
            
    return coo_matrix((data, (row, col)), shape=(num_constraints, num_vars))

def run_packing():
    n = 26
    
    best_result = None
    best_sum_r = -1.0
    
    # Define bounds
    bounds = [(0, 1) if k % 3 != 2 else (0, 0.5) for k in range(3*n)]
    
    # Try multiple initial configurations to avoid local optima
    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        centers, radii = generate_initial_config(n, seed=seed)
        
        # Flatten
        x0 = np.zeros(3*n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        
        # Define constraints for minimize
        cons = {
            'type': 'ineq',
            'fun': lambda v: constraints_fun(v, n),
            'jac': lambda v: constraints_jac(v, n).toarray()
        }
        
        # Optimize
        res = minimize(
            fun=lambda v: objective(v, n),
            x0=x0,
            method='SLSQP',
            jac=lambda v: objective_jac(v, n),
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success:
            sum_r = -res.fun
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_result = res.x.copy()
    
    if best_result is None:
        # Fallback if optimization failed
        centers, radii = generate_initial_config(n, seed=0)
        best_result = np.zeros(3*n)
        best_result[0::3] = centers[:, 0]
        best_result[1::3] = centers[:, 1]
        best_result[2::3] = radii
        best_sum_r = np.sum(radii)
        
    # Extract final centers and radii
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    final_centers[:, 0] = best_result[0::3]
    final_centers[:, 1] = best_result[1::3]
    final_radii = best_result[2::3]
    
    # Ensure non-negative radii (clipping to 0 if slightly negative due to numerical error)
    final_radii = np.maximum(final_radii, 0)
    
    return final_centers, final_radii, best_sum_r