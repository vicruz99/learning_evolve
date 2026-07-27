import numpy as np
from scipy.optimize import minimize

def compute_energy(centers, r):
    n = centers.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            if dist < 2.0 * r:
                overlap = 2.0 * r - dist
                energy += overlap * overlap
        x = centers[i, 0]
        y = centers[i, 1]
        if x < r:
            energy += (r - x)**2
        elif x > 1.0 - r:
            energy += (x - (1.0 - r))**2
        if y < r:
            energy += (r - y)**2
        elif y > 1.0 - r:
            energy += (y - (1.0 - r))**2
    return energy

def compute_gradient(centers, r):
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            if dist < 2.0 * r and dist > 1e-8:
                overlap = 2.0 * r - dist
                factor = 2.0 * overlap / dist
                grad[i, 0] += factor * dx
                grad[i, 1] += factor * dy
                grad[j, 0] -= factor * dx
                grad[j, 1] -= factor * dy
        x = centers[i, 0]
        y = centers[i, 1]
        if x < r:
            grad[i, 0] += 2.0 * (r - x)
        elif x > 1.0 - r:
            grad[i, 0] -= 2.0 * (x - (1.0 - r))
        if y < r:
            grad[i, 1] += 2.0 * (r - y)
        elif y > 1.0 - r:
            grad[i, 1] -= 2.0 * (y - (1.0 - r))
    return grad

def objective_wrapper(c_flat, r, n):
    return compute_energy(c_flat.reshape(n, 2), r)

def gradient_wrapper(c_flat, r, n):
    return compute_gradient(c_flat.reshape(n, 2), r).flatten()

def generate_initial_centers(n, mode='grid'):
    if mode == 'grid':
        centers_list = []
        for i in range(6):
            y = 0.1 + i * 0.15
            offset = 0.075 if i % 2 == 1 else 0.0
            for j in range(5):
                x = 0.1 + j * 0.18 + offset
                if 0 <= x <= 1 and 0 <= y <= 1:
                    centers_list.append([x, y])
                if len(centers_list) >= n:
                    break
            if len(centers_list) >= n:
                break
        return np.array(centers_list[:n])
    else:
        np.random.seed(42)
        return np.random.uniform(0.1, 0.9, size=(n, 2))

def run_single_packing(n, init_centers):
    best_r = 0.05
    best_centers = init_centers.copy()
    current_centers = init_centers.copy()
    r = 0.05
    step = 0.001
    tol = 1e-7
    bounds = [(0.0, 1.0)] * (2 * n)

    max_r_limit = 0.11
    while r < max_r_limit:
        res = minimize(
            fun=objective_wrapper,
            args=(r, n),
            x0=current_centers.flatten(),
            jac=gradient_wrapper,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if res.fun < tol:
            current_centers = res.x.reshape(n, 2)
            best_r = r
            best_centers = current_centers.copy()
            r += step
        else:
            step /= 2.0
            if step < 1e-6:
                break
            current_centers += np.random.normal(0, 0.005, size=current_centers.shape)
            current_centers = np.clip(current_centers, 0.001, 0.999)
            continue

    res = minimize(
        fun=objective_wrapper,
        args=(best_r, n),
        x0=best_centers.flatten(),
        jac=gradient_wrapper,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 1500}
    )
    final_centers = res.x.reshape(n, 2)
    return final_centers, best_r

def run_packing():
    n = 26
    centers1 = generate_initial_centers(n, mode='grid')
    res1_centers, r1 = run_single_packing(n, centers1)
    
    centers2 = generate_initial_centers(n, mode='random')
    res2_centers, r2 = run_single_packing(n, centers2)
    
    if r1 >= r2:
        final_centers = res1_centers
        best_r = r1
    else:
        final_centers = res2_centers
        best_r = r2
        
    radii = np.full(n, best_r)
    sum_radii = np.sum(radii)
    return final_centers, radii, sum_radii