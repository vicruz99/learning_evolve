# sol_000275 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=8e9c3de1 sum of radii=2.616307 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_loss_and_grad(params, n, mu):
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    loss = -np.sum(radii)
    grad = np.zeros_like(params)
    grad[2*n:] = -1.0
    
    # Boundary penalties and gradients
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        v = r - x
        if v > 1e-8:
            loss += mu * v**2
            grad[2*n + i] += mu * 2 * v
            grad[2*i] -= mu * 2 * v
            
        v = x + r - 1
        if v > 1e-8:
            loss += mu * v**2
            grad[2*n + i] += mu * 2 * v
            grad[2*i] += mu * 2 * v
            
        v = r - y
        if v > 1e-8:
            loss += mu * v**2
            grad[2*n + i] += mu * 2 * v
            grad[2*i + 1] -= mu * 2 * v
            
        v = y + r - 1
        if v > 1e-8:
            loss += mu * v**2
            grad[2*n + i] += mu * 2 * v
            grad[2*i + 1] += mu * 2 * v

    # Overlap penalties and gradients
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
                
            min_dist = radii[i] + radii[j]
            v = min_dist - dist
            if v > 1e-8:
                loss += mu * v**2
                grad[2*n + i] += mu * 2 * v
                grad[2*n + j] += mu * 2 * v
                
                coef = -mu * 2 * v / dist
                grad[2*i] += coef * dx
                grad[2*i + 1] += coef * dy
                grad[2*j] -= coef * dx
                grad[2*j + 1] -= coef * dy
                
    return loss, grad

def obj_func(x, mu):
    return compute_loss_and_grad(x, 26, mu)[0]

def jac_func(x, mu):
    return compute_loss_and_grad(x, 26, mu)[1]

def generate_hex_config(n):
    row_counts = [5, 6, 5, 6, 4]
    centers = []
    for idx, count in enumerate(row_counts):
        y = 0.12 + idx * 0.19
        x_start = (1.0 - (count-1)*0.14)/2
        for k in range(count):
            x = x_start + k * 0.14
            if idx % 2 == 1:
                x += 0.07
            centers.append([x, y])
    while len(centers) < n:
        centers.append([0.5, 0.5])
    return np.array(centers[:n])

def generate_random_config(n):
    np.random.seed(123)
    centers = np.random.uniform(0.12, 0.88, (n, 2))
    return centers

def solve_radii_lp(centers, n):
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    b_i = np.array([min(c[0], 1-c[0], c[1], 1-c[1]) for c in centers])
    b_i = np.clip(b_i, 1e-7, 0.5)
    
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    bounds = [(0.0, bi) for bi in b_i]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=np.array(b_ub), bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 0.05)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    configs = [generate_hex_config(n), generate_random_config(n)]
    best_sum_r = 0.0
    best_centers = None
    best_radii = None
    
    for init_centers in configs:
        init_radii = np.full(n, 0.09)
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        
        mu = 2.0
        x_curr = x0
        for _ in range(6):
            mu *= 2.5
            res = minimize(obj_func, x_curr, jac=jac_func, args=(mu,), 
                           method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 400, 'ftol': 1e-10})
            x_curr = res.x
            
        centers_opt = x_curr[:2*n].reshape(n, 2)
        radii_opt = solve_radii_lp(centers_opt, n)
        
        current_sum = np.sum(radii_opt)
        if current_sum > best_sum_r:
            best_sum_r = current_sum
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
            
    # Final safety clamping
    for i in range(n):
        best_radii[i] = min(best_radii[i], best_centers[i,0], 1-best_centers[i,0], 
                            best_centers[i,1], 1-best_centers[i,1])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            r_sum = best_radii[i] + best_radii[j]
            if r_sum > d - 1e-9:
                factor = (d - 1e-9) / r_sum
                best_radii[i] *= factor
                best_radii[j] *= factor
                
    return best_centers, best_radii, np.sum(best_radii)
