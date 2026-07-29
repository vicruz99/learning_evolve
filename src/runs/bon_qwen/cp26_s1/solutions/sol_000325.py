# sol_000325 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fe3e1745) state=f042dddc sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import itertools

def solve_lp_radii(centers):
    """
    Given fixed centers, solve the LP to find optimal radii.
    Maximize sum(r) subject to r_i + r_j <= dist(c_i, c_j) and boundary constraints.
    """
    n = centers.shape[0]
    c = np.array([1.0] * n)  # Objective: maximize sum(r)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = [0.0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1-x
        row = [0.0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - x)
        
        # r_i <= y
        row = [0.0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1-y
        row = [0.0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - y)

    # Non-overlap constraints: r_i + r_j <= dist(c_i, c_j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = [0.0] * n
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)

    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback to small radii if LP fails
        return np.ones(n) * 0.01

def objective_function(centers_flat):
    """
    Objective function for center optimization.
    Returns negative sum of radii.
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    
    # Clip centers to valid range to avoid numerical issues in LP
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    
    radii = solve_lp_radii(centers)
    return -np.sum(radii), radii

def get_initial_centers_hexagonal(n):
    """
    Generate initial centers using a hexagonal packing pattern.
    """
    centers = []
    # Approximate grid dimensions
    # For 26 circles, roughly 5x5 or 6x4
    # Let's try to pack them in a hexagonal lattice
    # Horizontal spacing 2r, vertical spacing sqrt(3)r
    # Assuming r approx 0.1, spacing approx 0.2 and 0.173
    
    # Let's try a grid of 6x5 (30 spots) and pick best 26 or just place 26
    rows = 6
    cols = 5
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n:
                break
            # Hexagonal shift for alternating rows
            x = (c + 1) * dx
            y = (r + 1) * dy
            if r % 2 == 1:
                x += dx / 2.0
            # Ensure within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
            count += 1
        if count >= n:
            break
            
    return np.array(centers)

def relax_centers(centers, radii, steps=500, step_size=0.01):
    """
    Force-directed relaxation to separate overlapping or tight circles.
    """
    n = centers.shape[0]
    for step in range(steps):
        forces = np.zeros_like(centers)
        current_sum_radii = np.sum(radii)
        
        # Calculate pairwise repulsive forces based on tightness
        # If r_i + r_j is close to dist, push apart
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                r_sum = radii[i] + radii[j]
                
                if dist > 1e-9:
                    # Overlap amount (positive if overlapping)
                    overlap = r_sum - dist
                    # We want to push apart if overlap > 0 or if very tight
                    # Force proportional to overlap
                    if overlap > -0.01: # Consider tight contacts
                        force_mag = overlap * 0.5 
                        direction = centers[i] - centers[j]
                        direction = direction / dist
                        forces[i] += direction * force_mag
                        forces[j] -= direction * force_mag
        
        # Boundary repulsion (push towards center if touching boundary)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0.001: forces[i, 0] += (0.001 - (x - r)) * 2.0
            if x + r > 0.999: forces[i, 0] -= ((x + r) - 0.999) * 2.0
            if y - r < 0.001: forces[i, 1] += (0.001 - (y - r)) * 2.0
            if y + r > 0.999: forces[i, 1] -= ((y + r) - 0.999) * 2.0
            
        # Update centers
        centers += forces * step_size
        centers = np.clip(centers, 0.001, 0.999)
        
        # Recompute radii
        radii = solve_lp_radii(centers)
        
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Generate initial centers
    n = 26
    centers = get_initial_centers_hexagonal(n)
    
    # 2. Solve for initial radii
    radii = solve_lp_radii(centers)
    
    # 3. Optimize centers using Nelder-Mead
    # We optimize the flat array of centers
    x0 = centers.flatten()
    
    # Using a simple iterative improvement with random restarts might be better than pure Nelder-Mead
    # due to the non-smooth nature of the objective.
    # Let's try Nelder-Mead first.
    
    best_val = -np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Nelder-Mead optimization
    # We define a function that returns the scalar objective
    def objective_scalar(x):
        obj, _ = objective_function(x)
        return obj

    # Run optimization
    # nelder-mead might struggle with 52 dims, but let's try
    res = minimize(objective_scalar, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    if res.fun < best_val:
        best_val = res.fun
        best_centers = res.x.reshape((n, 2))
        _, best_radii = objective_function(res.x)
        
    # 4. Relaxation step to fine-tune
    centers_relaxed, radii_relaxed = relax_centers(best_centers, best_radii, steps=1000, step_size=0.005)
    
    final_sum = np.sum(radii_relaxed)
    
    # Validate and return
    if np.sum(radii_relaxed) > best_val * -1:
        return centers_relaxed, radii_relaxed, np.sum(radii_relaxed)
    
    return best_centers, best_radii, -best_val

# Helper to run and print stats if called directly (not part of required function)
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radii: {r}")
