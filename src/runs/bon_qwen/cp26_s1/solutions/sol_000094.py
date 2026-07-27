# sol_000094 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5526c41b) state=51666437 sum of radii=2.318475 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize

def calculate_score(centers, radii):
    """
    Calculates the score (sum of radii minus penalties) for a given configuration.
    """
    n = centers.shape[0]
    score_val = np.sum(radii)
    penalty = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Boundary penalties
        if x - r < 0: penalty += 1000 * (r - x)**2
        if x + r > 1: penalty += 1000 * (x + r - 1)**2
        if y - r < 0: penalty += 1000 * (r - y)**2
        if y + r > 1: penalty += 1000 * (y + r - 1)**2
        
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 1e-6:
                penalty += 1000 * overlap**2
    return score_val - penalty

def optimize_radii_lp(centers):
    """
    Solves a Linear Programming problem to find the maximum sum of radii
    for fixed center positions.
    """
    n = centers.shape[0]
    A = []
    b = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    for i in range(n):
        x, y = centers[i]
        row_x = np.zeros(n)
        row_x[i] = 1
        A.append(row_x)
        b.append(x)
        
        A.append(row_x)
        b.append(1-x)
        
        row_y = np.zeros(n)
        row_y[i] = 1
        A.append(row_y)
        b.append(y)
        
        A.append(row_y)
        b.append(1-y)
        
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A.append(row)
            b.append(dist)
            
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c_obj = -np.ones(n)
    bounds = [(0, None) for _ in range(n)]
    
    try:
        # Use HiGHS solver for robustness
        res = scipy.optimize.linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to pack 26 circles in a unit square.
    """
    n = 26
    np.random.seed(42)
    
    # Initialize centers on a perturbed grid
    centers = np.zeros((n, 2))
    count = 0
    for i in range(5):
        for j in range(5):
            if count < n:
                centers[count, 0] = 0.1 + i * 0.2 + (np.random.rand()-0.5)*0.01
                centers[count, 1] = 0.1 + j * 0.2 + (np.random.rand()-0.5)*0.01
                count += 1
    if count < n:
        centers[count, 0] = 0.5 + (np.random.rand()-0.5)*0.05
        centers[count, 1] = 0.5 + (np.random.rand()-0.5)*0.05
        count += 1
        
    # Initial radii
    radii = np.ones(n) * 0.05
    
    # Simulated Annealing state
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = calculate_score(centers, radii)
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    current_score = best_score
    
    temp = 0.1
    temp_min = 1e-4
    cooling = 0.995
    step_size = 0.05
    
    iterations = 30000
    
    for i in range(iterations):
        idx = np.random.randint(n)
        
        # Propose new center
        new_centers = current_centers.copy()
        new_centers[idx] = np.clip(new_centers[idx] + np.random.randn(2) * step_size, 0, 1)
        
        # Propose new radius
        new_radii = current_radii.copy()
        new_radii[idx] = new_radii[idx] + np.random.randn() * step_size * 0.5
        if new_radii[idx] < 0:
            new_radii[idx] = 0
        
        new_score = calculate_score(new_centers, new_radii)
        
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta > 0 or np.random.rand() < math.exp(delta / temp):
            current_centers = new_centers
            current_radii = new_radii
            current_score = new_score
            
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_radii = current_radii.copy()
                best_score = current_score
        
        temp *= cooling
        if temp < temp_min:
            temp = temp_min
            
    # Refine radii using Linear Programming for the best centers found
    final_radii = optimize_radii_lp(best_centers)
    if final_radii is not None and not np.any(np.isnan(final_radii)):
        final_sum = np.sum(final_radii)
        return best_centers, final_radii, final_sum
    else:
        return best_centers, best_radii, best_score
