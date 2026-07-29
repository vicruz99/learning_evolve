# sol_000211 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 722eaafb) state=24299820 sum of radii=2.600952 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def get_optimal_radii(centers):
    """
    Solves the LP to maximize sum of radii given fixed centers.
    Returns (radii, sum_radii).
    """
    n = len(centers)
    if n == 0:
        return np.array([]), 0.0

    # Objective: Maximize sum(r) => Minimize -sum(r)
    c = -np.ones(n)
    
    # Bounds: 0 <= r_i <= wall_distance
    bounds = []
    for i in range(n):
        x, y = centers[i]
        # Distance to nearest wall
        w = min(x, 1.0 - x, y, 1.0 - y)
        # Ensure non-negative bound
        w = max(w, 0.0)
        bounds.append((0.0, w))
        
    # Constraints: r_i + r_j <= dist(i, j)
    num_pairs = n * (n - 1) // 2
    
    if num_pairs > 0:
        A_ub = np.zeros((num_pairs, n))
        b_ub = np.zeros(num_pairs)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub[idx] = row
                b_ub[idx] = dist
                idx += 1
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x, -res.fun
        except Exception:
            pass
    
    # Fallback
    radii = np.array([min(x, 1.0-x, y, 1.0-y) for x, y in centers])
    return radii, np.sum(radii)

def run_packing():
    n = 26
    
    # --- Initialization ---
    # Create a hexagonal-ish grid initialization
    cols = 5
    rows = 5
    # Create points in a 5x5 grid
    x_coords = np.linspace(0.1, 0.9, cols)
    y_coords = np.linspace(0.1, 0.9, rows)
    
    centers_list = []
    for y in y_coords:
        for x in x_coords:
            centers_list.append([x, y])
    
    # We have 25 points. Add 1 more.
    # Place it in the middle of a cell or slightly offset to break symmetry
    centers_list.append([0.25, 0.25]) 
    
    # Convert to numpy array
    centers = np.array(centers_list[:n])
    
    # Add small random noise to break symmetry and help optimization
    np.random.seed(42) # For reproducibility
    noise = np.random.uniform(-0.005, 0.005, size=centers.shape)
    centers = centers + noise
    # Clip to keep inside valid range loosely
    centers = np.clip(centers, 0.01, 0.99)

    # --- Optimization Loop ---
    num_iterations = 300
    step_size = 0.02
    
    best_sum = 0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    for iteration in range(num_iterations):
        # 1. Solve for optimal radii
        radii, current_sum = get_optimal_radii(centers)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
        
        # 2. Compute forces
        forces = np.zeros((n, 2))
        tolerance = 1e-6
        repulsion_strength = 0.8
        wall_repulsion = 0.8
        
        # Pair interactions
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r + tolerance:
                    if dist > 1e-9:
                        fx = dx / dist
                        fy = dy / dist
                    else:
                        fx = np.random.uniform(-1, 1)
                        fy = np.random.uniform(-1, 1)
                        norm = math.sqrt(fx*fx + fy*fy)
                        if norm > 1e-9:
                            fx /= norm
                            fy /= norm
                        else:
                            fx, fy = 1.0, 0.0
                    
                    overlap = max(0.0, sum_r - dist)
                    # Force magnitude
                    force_mag = repulsion_strength * (1.0 + overlap * 15.0)
                    
                    forces[i][0] += fx * force_mag
                    forces[i][1] += fy * force_mag
                    forces[j][0] -= fx * force_mag
                    forces[j][1] -= fy * force_mag
        
        # Wall interactions
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left
            if x < r + tolerance:
                forces[i][0] += wall_repulsion * (1.0 + (r - x) * 15.0)
            # Right
            if x > 1.0 - r - tolerance:
                forces[i][0] -= wall_repulsion * (1.0 + (x - (1.0 - r)) * 15.0)
            # Bottom
            if y < r + tolerance:
                forces[i][1] += wall_repulsion * (1.0 + (r - y) * 15.0)
            # Top
            if y > 1.0 - r - tolerance:
                forces[i][1] -= wall_repulsion * (1.0 + (y - (1.0 - r)) * 15.0)

        # 3. Update centers
        new_centers = centers + step_size * forces
        new_centers = np.clip(new_centers, 1e-5, 1.0 - 1e-5)
        centers = new_centers
        
        # Decay step size
        step_size *= 0.99

    # Final evaluation
    final_radii, final_sum = get_optimal_radii(best_centers)
    
    # Check current centers just in case
    cur_radii, cur_sum = get_optimal_radii(centers)
    if cur_sum > final_sum:
        final_sum = cur_sum
        final_radii = cur_radii
        best_centers = centers

    return best_centers, final_radii, float(final_sum)
