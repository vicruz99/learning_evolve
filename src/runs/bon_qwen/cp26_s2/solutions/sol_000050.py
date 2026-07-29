# sol_000050 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 79449191) state=50255595 sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_and_grad(x_vec, n):
    cx = x_vec[0::3]
    cy = x_vec[1::3]
    cr = x_vec[2::3]
    
    val, grad = calculate_val_and_grad(cx, cy, cr)
    return val, grad

def calculate_val_and_grad(cx, cy, cr):
    n = len(cx)
    sum_r = np.sum(cr)
    obj = -sum_r
    
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0
    
    P_overlap = 1000.0
    P_boundary = 1000.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            
            if d < 1e-12:
                d = 1e-12
                dx = 1e-12
                dy = 0
            
            overlap = cr[i] + cr[j] - d
            if overlap > 0:
                obj += P_overlap * overlap**2
                term = 2 * P_overlap * overlap
                
                grad[3*i + 2] += term
                grad[3*j + 2] += term
                
                factor_x = -term * (dx / d)
                grad[3*i] += factor_x
                grad[3*j] -= factor_x
                
                factor_y = -term * (dy / d)
                grad[3*i + 1] += factor_y
                grad[3*j + 1] -= factor_y

    for i in range(n):
        viol = cr[i] - cx[i]
        if viol > 0:
            obj += P_boundary * viol**2
            term = 2 * P_boundary * viol
            grad[3*i + 2] += term
            grad[3*i] -= term
            
        viol = cr[i] + cx[i] - 1.0
        if viol > 0:
            obj += P_boundary * viol**2
            term = 2 * P_boundary * viol
            grad[3*i + 2] += term
            grad[3*i] += term
                
        viol = cr[i] - cy[i]
        if viol > 0:
            obj += P_boundary * viol**2
            term = 2 * P_boundary * viol
            grad[3*i + 2] += term
            grad[3*i + 1] -= term
                
        viol = cr[i] + cy[i] - 1.0
        if viol > 0:
            obj += P_boundary * viol**2
            term = 2 * P_boundary * viol
            grad[3*i + 2] += term
            grad[3*i + 1] += term
                
    return obj, grad

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    for seed in range(30):
        np.random.seed(seed * 12345)
        
        centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
        radii = np.ones(n_circles) * 0.02
        
        x0 = np.zeros(3 * n_circles)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        
        bounds = [(0, 1)] * (2 * n_circles) + [(0, 1)] * n_circles
        
        res = minimize(lambda x: objective_and_grad(x, n_circles), x0, jac=True, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        c_x = res.x[0::3]
        c_y = res.x[1::3]
        c_r = res.x[2::3]
        
        # Validation check
        valid = True
        for i in range(n_circles):
            if c_x[i] < c_r[i] - 1e-12 or c_x[i] > 1 - c_r[i] + 1e-12: valid = False
            if c_y[i] < c_r[i] - 1e-12 or c_y[i] > 1 - c_r[i] + 1e-12: valid = False
            for j in range(i + 1, n_circles):
                dist = np.sqrt((c_x[i] - c_x[j])**2 + (c_y[i] - c_y[j])**2)
                if dist < c_r[i] + c_r[j] - 1e-12:
                    valid = False
                    break
            if not valid: break
        
        if valid:
            current_sum = np.sum(c_r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = np.column_stack((c_x, c_y))
                best_radii = c_r

    if best_centers is None:
        centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
        radii = np.ones(n_circles) * 0.01
        best_sum = np.sum(radii)
        best_centers = centers
        best_radii = radii
            
    return best_centers, best_radii, best_sum
