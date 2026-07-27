import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def neg_sum_radii(params):
    """Objective: minimize negative sum of radii"""
    radii = params[2 * N_CIRCLES:]
    return -np.sum(radii)

def compute_constraints(params):
    """Compute all constraint values that must be >= 0"""
    centers = params[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = params[2 * N_CIRCLES:]
    
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(N_CIRCLES):
        x, y = centers[i]
        r = radii[i]
        constraints.append(x - r)
        constraints.append(1.0 - x - r)
        constraints.append(y - r)
        constraints.append(1.0 - y - r)
        constraints.append(r)
    
    # Non-overlap constraints for each pair
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            constraints.append(dist - radii[i] - radii[j])
    
    return np.array(constraints)

def generate_grid_init(rng, n_circles):
    """Generate grid-based initial configuration"""
    centers = np.zeros((n_circles, 2))
    idx = 0
    for row in range(6):
        for col in range(5):
            if idx < n_circles:
                centers[idx, 0] = 0.1 + col * 0.18
                centers[idx, 1] = 0.1 + row * 0.18
                centers[idx, 0] += rng.uniform(-0.03, 0.03)
                centers[idx, 1] += rng.uniform(-0.03, 0.03)
                centers[idx] = np.clip(centers[idx], 0.05, 0.95)
                idx += 1
    while idx < n_circles:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    return centers

def generate_hex_init(rng, n_circles):
    """Generate hexagonal lattice initial configuration"""
    centers = np.zeros((n_circles, 2))
    idx = 0
    spacing = 0.14
    for row in range(8):
        offset = spacing / 2 if row % 2 == 1 else 0
        for col in range(8):
            if idx >= n_circles:
                break
            x = offset + col * spacing
            y = row * spacing * 0.8660254037844386
            if x <= 0.95 and y <= 0.95:
                centers[idx, 0] = x + rng.uniform(-0.02, 0.02)
                centers[idx, 1] = y + rng.uniform(-0.02, 0.02)
                centers[idx] = np.clip(centers[idx], 0.05, 0.95)
                idx += 1
        if idx >= n_circles:
            break
    while idx < n_circles:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    return centers

def generate_random_init(rng, n_circles):
    """Generate random initial configuration"""
    centers = rng.uniform(0.1, 0.9, (n_circles, 2))
    return centers

def run_optimization(centers, radii, max_iter, ftol):
    """Run single optimization trial"""
    params = np.concatenate([centers.ravel(), radii])
    constraint = {'type': 'ineq', 'fun': compute_constraints}
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    
    result = minimize(
        neg_sum_radii,
        params,
        method='SLSQP',
        bounds=bounds,
        constraints=[constraint],
        options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
    )
    
    return result

def run_packing():
    best_sum = 0.0
    best_params = None
    
    init_methods = [generate_grid_init, generate_hex_init, generate_random_init]
    
    # Phase 1: Broad search with many restarts
    for trial in range(80):
        rng = np.random.RandomState(trial * 7 + 13)
        
        # Cycle through initialization methods
        method_idx = trial % len(init_methods)
        centers = init_methods[method_idx](rng, N_CIRCLES)
        radii = np.ones(N_CIRCLES) * 0.04
        
        result = run_optimization(centers, radii, max_iter=5000, ftol=1e-14)
        
        current_sum = np.sum(result.x[2 * N_CIRCLES:])
        if current_sum > best_sum:
            best_sum = current_sum
            best_params = result.x.copy()
    
    # Phase 2: Refinement from best found solutions
    if best_params is not None:
        for trial in range(30):
            rng = np.random.RandomState(trial * 3 + 7)
            
            # Perturb best solution
            perturbed = best_params.copy()
            perturbation = rng.uniform(-0.005, 0.005, size=perturbed.shape)
            perturbed += perturbation
            perturbed[:2 * N_CIRCLES] = np.clip(perturbed[:2 * N_CIRCLES], 0.0, 1.0)
            perturbed[2 * N_CIRCLES:] = np.clip(perturbed[2 * N_CIRCLES:], 0.0, 0.5)
            
            result = run_optimization(
                perturbed[:2 * N_CIRCLES].reshape(N_CIRCLES, 2),
                perturbed[2 * N_CIRCLES:],
                max_iter=10000,
                ftol=1e-16
            )
            
            current_sum = np.sum(result.x[2 * N_CIRCLES:])
            if current_sum > best_sum:
                best_sum = current_sum
                best_params = result.x.copy()
    
    # Extract final solution
    centers = best_params[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_params[2 * N_CIRCLES:]
    
    return centers, radii, best_sum