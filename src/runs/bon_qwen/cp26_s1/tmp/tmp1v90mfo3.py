import numpy as np
from scipy.optimize import linprog, minimize, basinhopping


def get_optimal_radii(centers):
    """Given fixed centers, solve LP to find optimal radii maximizing sum."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub_list = []
    b_ub_list = []
    
    for i in range(n):
        x, y = centers[i]
        for bound in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(bound)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(d)
    
    bounds = [(0.0, None)] * n
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        return result.x
    else:
        return compute_radii_greedy(centers)


def compute_radii_greedy(centers):
    """Fallback: compute radii greedily when LP fails."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_r = min(max_r, d - radii[j])
        radii[i] = max(0.0, max_r)
    return radii


def objective_function(centers_flat):
    """Negative sum of radii (we minimize this)."""
    centers = centers_flat.reshape(26, 2)
    radii = get_optimal_radii(centers)
    return -np.sum(radii)


def make_hexagonal_centers(r_init, jitter_magnitude=0.0):
    """Create a hexagonal grid of 26 circle centers."""
    centers = np.zeros((26, 2))
    row_spacing = r_init * np.sqrt(3)
    col_spacing = 2.0 * r_init
    
    idx = 0
    row_pattern = [5, 5, 5, 5, 5, 1]
    
    for row in range(6):
        n_in_row = row_pattern[row]
        offset = r_init if row % 2 == 1 else 0.0
        for col in range(n_in_row):
            x = 0.5 + (col - (n_in_row - 1) / 2.0) * col_spacing + offset
            y = 0.5 + (row - 2.5) * row_spacing
            centers[idx] = np.array([x, y])
            idx += 1
    
    if jitter_magnitude > 0:
        jitter = np.random.uniform(-jitter_magnitude, jitter_magnitude, (26, 2))
        centers = centers + jitter
    
    centers = np.clip(centers, 0.005, 0.995)
    return centers


def step_func(x, stepsize):
    """Step function for basin hopping."""
    new_x = x + np.random.uniform(-stepsize, stepsize, size=x.shape)
    new_centers = new_x.reshape(26, 2)
    new_centers = np.clip(new_centers, 0.005, 0.995)
    return new_centers.flatten()


def run_packing():
    np.random.seed(123)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Multiple hexagonal starts with Nelder-Mead
    for trial in range(30):
        r_init = 0.101 + np.random.uniform(-0.003, 0.003)
        jitter_mag = 0.02 + np.random.uniform(0, 0.03)
        centers = make_hexagonal_centers(r_init, jitter_mag)
        
        result = minimize(
            objective_function,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        opt_centers = result.x.reshape(26, 2)
        opt_radii = get_optimal_radii(opt_centers)
        current_sum = np.sum(opt_radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_centers.copy()
            best_radii = opt_radii.copy()
    
    # Phase 2: Basin hopping from best solution found
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-12}
    }
    
    ret = basinhopping(
        objective_function,
        best_centers.flatten(),
        minimizer_kwargs=minimizer_kwargs,
        stepsize=0.003,
        niter=200,
        accept_test=None,
        take_step=step_func
    )
    
    opt_centers = ret.x.reshape(26, 2)
    opt_radii = get_optimal_radii(opt_centers)
    current_sum = np.sum(opt_radii)
    
    if current_sum > best_sum:
        best_sum = current_sum
        best_centers = opt_centers.copy()
        best_radii = opt_radii.copy()
    
    # Phase 3: Additional fine-tuning from multiple perturbed copies of best
    for trial in range(10):
        perturbed = best_centers.copy()
        perturbed += np.random.uniform(-0.005, 0.005, (26, 2))
        perturbed = np.clip(perturbed, 0.005, 0.995)
        
        result = minimize(
            objective_function,
            perturbed.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 30000, 'xatol': 1e-11, 'fatol': 1e-12}
        )
        
        opt_centers = result.x.reshape(26, 2)
        opt_radii = get_optimal_radii(opt_centers)
        current_sum = np.sum(opt_radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_centers.copy()
            best_radii = opt_radii.copy()
    
    # Phase 4: Try alternative row patterns
    for trial in range(15):
        r_init = 0.101 + np.random.uniform(-0.002, 0.002)
        centers = np.zeros((26, 2))
        row_spacing = r_init * np.sqrt(3)
        col_spacing = 2.0 * r_init
        
        # Try different patterns: 6-4-6-4-6 or 5-5-5-5-6 etc
        pattern_choice = trial % 4
        if pattern_choice == 0:
            row_pattern = [6, 4, 6, 4, 6]  # 26
        elif pattern_choice == 1:
            row_pattern = [5, 5, 5, 6, 5]  # 26
        elif pattern_choice == 2:
            row_pattern = [4, 6, 4, 6, 6]  # 26
        else:
            row_pattern = [5, 6, 5, 5, 5]  # 26
        
        idx = 0
        total_rows = len(row_pattern)
        for row in range(total_rows):
            n_in_row = row_pattern[row]
            offset = r_init if row % 2 == 1 else 0.0
            for col in range(n_in_row):
                x = 0.5 + (col - (n_in_row - 1) / 2.0) * col_spacing + offset
                y = 0.5 + (row - (total_rows - 1) / 2.0) * row_spacing
                centers[idx] = np.array([x, y])
                idx += 1
        
        jitter = np.random.uniform(-0.015, 0.015, (26, 2))
        centers = centers + jitter
        centers = np.clip(centers, 0.005, 0.995)
        
        result = minimize(
            objective_function,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 40000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        opt_centers = result.x.reshape(26, 2)
        opt_radii = get_optimal_radii(opt_centers)
        current_sum = np.sum(opt_radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_centers.copy()
            best_radii = opt_radii.copy()
    
    return best_centers, best_radii, float(best_sum)