import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[2::3])

def constraint(vars):
    """Inequality constraints: g(vars) >= 0"""
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    dist_sq = dx**2 + dy**2
    cons = dist_sq - dr**2
    
    idx = np.triu_indices(n, k=1)
    c = np.concatenate([c, cons[idx]])
    return c

def constraint_jac(vars):
    """Analytic Jacobian of constraints."""
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    num_constraints = 4*n + n*(n-1)//2
    jac = np.zeros((num_constraints, 3*n))
    
    # Boundary constraint gradients
    jac[0:n, 0::3] = 1.0
    jac[0:n, 2::3] = -1.0
    
    jac[n:2*n, 0::3] = -1.0
    jac[n:2*n, 2::3] = -1.0
    
    jac[2*n:3*n, 1::3] = 1.0
    jac[2*n:3*n, 2::3] = -1.0
    
    jac[3*n:4*n, 1::3] = -1.0
    jac[3*n:4*n, 2::3] = -1.0
    
    # Overlap constraint gradients
    row = 4*n
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            
            jac[row, 3*i] = 2.0 * dx
            jac[row, 3*j] = -2.0 * dx
            jac[row, 3*i+1] = 2.0 * dy
            jac[row, 3*j+1] = -2.0 * dy
            jac[row, 3*i+2] = -2.0 * dr
            jac[row, 3*j+2] = -2.0 * dr
            row += 1
    return jac

def run_packing():
    n = 26
    bounds = [(0.0, 1.0) for _ in range(3*n)]
    cons = {'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}
    
    best_sum = -1.0
    best_vars = None
    
    # Structured grid initialization
    xs = np.linspace(0.15, 0.85, 6)
    ys = np.linspace(0.15, 0.85, 5)
    base_pts = np.array([(x, y) for y in ys for x in xs])
    
    # Multi-start optimization
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts = base_pts[:n].copy()
        pts += rng.uniform(-0.03, 0.03, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        vars0 = np.zeros(3*n)
        for i in range(n):
            vars0[3*i] = pts[i, 0]
            vars0[3*i+1] = pts[i, 1]
            vars0[3*i+2] = 0.07  # Initial radius guess
            
        res = minimize(objective, vars0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-9})
        
        if res.success:
            current_sum = np.sum(res.x[2::3])
            if current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x
                
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_vars[3*i]
        centers[i, 1] = best_vars[3*i+1]
        radii[i] = max(0.0, best_vars[3*i+2])
        
    return centers, radii, np.sum(radii)