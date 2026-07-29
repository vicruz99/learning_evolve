# sol_000144 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 68244382) state=de69a397 sum of radii=0.921482 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
BOUNDS = [(0.0, 1.0) for _ in range(2*N_CIRCLES)] + [(1e-6, 0.5) for _ in range(N_CIRCLES)]

def compute_loss_and_grad(params, lam, N):
    centers = params[:2*N].reshape(N, 2)
    radii = params[2*N:]
    
    penalty = 0.0
    grad_penalty = np.zeros_like(params)
    
    # Boundary violations
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        
        # Left: x >= r
        viol = r - x
        if viol > 0:
            penalty += viol**2
            grad_penalty[2*N + i] += 2 * viol
            grad_penalty[i] -= 2 * viol
        # Right: x + r <= 1
        viol = x + r - 1.0
        if viol > 0:
            penalty += viol**2
            grad_penalty[2*N + i] += 2 * viol
            grad_penalty[i] += 2 * viol
        # Bottom: y >= r
        viol = r - y
        if viol > 0:
            penalty += viol**2
            grad_penalty[2*N + i] += 2 * viol
            grad_penalty[N + i] -= 2 * viol
        # Top: y + r <= 1
        viol = y + r - 1.0
        if viol > 0:
            penalty += viol**2
            grad_penalty[2*N + i] += 2 * viol
            grad_penalty[N + i] += 2 * viol

    # Pairwise overlap violations
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            viol = radii[i] + radii[j] - dist
            if viol > 0:
                penalty += viol**2
                grad_penalty[2*N + i] += 2 * viol
                grad_penalty[2*N + j] += 2 * viol
                if dist > 1e-9:
                    inv_dist = 1.0 / dist
                    grad_penalty[i] -= 2 * viol * dx * inv_dist
                    grad_penalty[j] += 2 * viol * dx * inv_dist
                    grad_penalty[N + i] -= 2 * viol * dy * inv_dist
                    grad_penalty[N + j] += 2 * viol * dy * inv_dist
                    
    # Objective: minimize -sum(r) + lambda * penalty
    loss = -np.sum(radii) + lam * penalty
    grad = np.zeros_like(params)
    grad[2*N:] -= 1.0  # Derivative of -sum(r) w.r.t r is -1
    grad += lam * grad_penalty
    
    return loss, grad

def objective_wrapper(params, lam, N):
    return compute_loss_and_grad(params, lam, N)

def generate_initial(N, seed):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = np.full(N, 0.08)
    return np.concatenate([centers.ravel(), radii])

def run_packing():
    N = 26
    current_params = generate_initial(N, 42)
    
    # Continuation schedule for penalty weight
    lambdas = [5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0]
    
    for lam in lambdas:
        best_res = None
        best_fun = np.inf
        
        # Create candidates: current best + random restarts
        candidates = [current_params]
        for s in range(2):
            candidates.append(generate_initial(N, int(lam) + s))
            
        for cand in candidates:
            res = minimize(objective_wrapper, cand, method='L-BFGS-B', 
                           bounds=BOUNDS, args=(lam, N), jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
                
        current_params = best_res.x.copy()
        
        # Add small perturbation to help escape local minima in next stage
        rng = np.random.RandomState(int(lam * 13))
        noise = rng.normal(0, 0.0005, size=current_params.shape)
        current_params += noise
        for i in range(len(BOUNDS)):
            current_params[i] = np.clip(current_params[i], BOUNDS[i][0], BOUNDS[i][1])
            
    # Final high-precision polish
    res_final = minimize(objective_wrapper, current_params, method='L-BFGS-B', 
                         bounds=BOUNDS, args=(10000.0, N), jac=True,
                         options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
    current_params = res_final.x
    
    centers = current_params[:2*N].reshape(N, 2)
    radii = current_params[2*N:]
    
    # Exact violation check and safety shrink
    max_viol = 1e-12
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: max_viol = max(max_viol, r - x)
        if x + r > 1: max_viol = max(max_viol, x + r - 1)
        if y - r < 0: max_viol = max(max_viol, r - y)
        if y + r > 1: max_viol = max(max_viol, y + r - 1)
        
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            v = radii[i] + radii[j] - d
            if v > max_viol: max_viol = v
            
    if max_viol > 1e-10:
        radii -= max_viol * 0.5
        radii = np.maximum(radii, 1e-7)

    sum_r = float(np.sum(radii))
    return centers, radii, sum_r
