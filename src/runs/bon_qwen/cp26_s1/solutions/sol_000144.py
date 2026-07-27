# sol_000144 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a66096c7) state=93f2aced sum of radii=2.539347 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_objective(params):
    radii = params[2 * N:]
    return -np.sum(radii)

def compute_boundary_constraints(params):
    centers = params[:2 * N].reshape(N, 2)
    radii = params[2 * N:]
    constraints = np.zeros(4 * N)
    for i in range(N):
        x = centers[i, 0]
        y = centers[i, 1]
        r = radii[i]
        constraints[4 * i] = x - r
        constraints[4 * i + 1] = 1 - x - r
        constraints[4 * i + 2] = y - r
        constraints[4 * i + 3] = 1 - y - r
    return constraints

def compute_overlap_constraints(params):
    centers = params[:2 * N].reshape(N, 2)
    radii = params[2 * N:]
    num_constraints = N * (N - 1) // 2
    constraints = np.zeros(num_constraints)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            constraints[idx] = dist - radii[i] - radii[j]
            idx += 1
    return constraints

def create_hexagonal_init(rows, cols, offset_rows, n_max):
    centers = []
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= n_max:
                break
            y = (row + 0.5) / rows
            x = (col + 0.5) / cols
            if row in offset_rows:
                x += 1.0 / (2 * cols)
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append([x, y])
        if len(centers) >= n_max:
            break
    return np.array(centers[:n_max])

def create_grid_init(n, rows, cols):
    centers = []
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= n:
                break
            y = (row + 0.5) / rows
            x = (col + 0.5) / cols
            centers.append([x, y])
        if len(centers) >= n:
            break
    return np.array(centers[:n])

def create_corner_optimized_init():
    centers = []
    
    # 4 corner circles
    margin = 0.22
    corners = [[margin, margin], [1-margin, margin], [margin, 1-margin], [1-margin, 1-margin]]
    centers.extend(corners)
    
    # Fill remaining with hexagonal-like pattern
    remaining = N - len(centers)
    
    # Middle rows
    for row in range(1, 5):
        y = (row + 0.5) / 6
        num_cols = 4
        for col in range(num_cols):
            if len(centers) >= N:
                break
            x = (col + 0.5) / 5 + (0.1 if row % 2 == 1 else 0)
            x = max(0.05, min(0.95, x))
            centers.append([x, y])
    
    # Additional circles in gaps
    gap_positions = [
        [0.3, 0.2], [0.7, 0.2], [0.3, 0.8], [0.7, 0.8],
        [0.5, 0.15], [0.5, 0.85],
        [0.15, 0.5], [0.85, 0.5],
        [0.25, 0.5], [0.75, 0.5],
        [0.4, 0.5], [0.6, 0.5]
    ]
    
    for pos in gap_positions:
        if len(centers) >= N:
            break
        centers.append(pos)
    
    return np.array(centers[:N])

def optimize_packing(initial_centers, initial_radius, max_iter=5000):
    centers = initial_centers
    radii = np.full(N, initial_radius)
    params = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.extend([(0.001, 0.999), (0.001, 0.999)])
        bounds.append((0.001, 0.5))
    
    constraints = [
        {'type': 'ineq', 'fun': compute_boundary_constraints},
        {'type': 'ineq', 'fun': compute_overlap_constraints}
    ]
    
    try:
        result = minimize(
            compute_objective,
            params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
        )
        return result.x
    except Exception:
        return params

def validate_and_fix(params):
    centers = params[:2 * N].reshape(N, 2)
    radii = params[2 * N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.001)
    
    # Ensure centers are within bounds considering radii
    for i in range(N):
        centers[i, 0] = max(radii[i] + 1e-8, min(1 - radii[i] - 1e-8, centers[i, 0]))
        centers[i, 1] = max(radii[i] + 1e-8, min(1 - radii[i] - 1e-8, centers[i, 1]))
    
    # Fix overlaps by reducing radii
    for iteration in range(100):
        max_violation = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                overlap = radii[i] + radii[j] - dist
                if overlap > max_violation:
                    max_violation = overlap
                    viol_i, viol_j = i, j
        
        if max_violation < 1e-10:
            break
        
        # Reduce the radii of overlapping circles
        reduction = max_violation / 2 + 1e-8
        radii[viol_i] = max(0.001, radii[viol_i] - reduction)
        radii[viol_j] = max(0.001, radii[viol_j] - reduction)
        
        # Re-clamp centers
        for i in range(N):
            centers[i, 0] = max(radii[i] + 1e-8, min(1 - radii[i] - 1e-8, centers[i, 0]))
            centers[i, 1] = max(radii[i] + 1e-8, min(1 - radii[i] - 1e-8, centers[i, 1]))
    
    return centers, radii

def run_packing():
    best_sum = -1
    best_centers = None
    best_radii = None
    
    # Try multiple initializations
    initializations = [
        # Hexagonal packing variations
        (create_hexagonal_init(5, 6, [1, 3], N), 0.07),
        (create_hexagonal_init(6, 5, [0, 2, 4], N), 0.07),
        (create_hexagonal_init(5, 6, [0, 2, 4], N), 0.065),
        (create_hexagonal_init(6, 5, [1, 3], N), 0.065),
        # Grid packing
        (create_grid_init(N, 5, 6), 0.065),
        (create_grid_init(N, 6, 5), 0.065),
        (create_grid_init(N, 7, 4), 0.06),
        (create_grid_init(N, 4, 7), 0.06),
        # Corner optimized
        (create_corner_optimized_init(), 0.05),
    ]
    
    for centers_init, r_init in initializations:
        params = optimize_packing(centers_init, r_init, max_iter=8000)
        centers, radii = validate_and_fix(params)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Refinement: try to expand radii iteratively
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    for sweep in range(500):
        improved = False
        for i in range(N):
            # Compute maximum possible radius for circle i
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            for j in range(N):
                if i != j:
                    dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                    max_r = min(max_r, dist - radii[j])
            
            # Try to expand
            new_r = min(max_r, radii[i] * 1.002)
            if new_r > radii[i] + 1e-10:
                old_r = radii[i]
                radii[i] = new_r
                
                # Adjust position to maintain feasibility
                best_pos = centers[i].copy()
                best_r = new_r
                
                # Try small adjustments
                for dx, dy in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                               (0.0007, 0.0007), (-0.0007, 0.0007), (0.0007, -0.0007), (-0.0007, -0.0007)]:
                    test_pos = centers[i] + np.array([dx, dy])
                    test_r = new_r
                    
                    # Check boundary
                    test_r = min(test_r, test_pos[0], 1 - test_pos[0], test_pos[1], 1 - test_pos[1])
                    
                    # Check overlaps
                    for j in range(N):
                        if i != j:
                            dist = np.sqrt((test_pos[0] - centers[j, 0])**2 + (test_pos[1] - centers[j, 1])**2)
                            test_r = min(test_r, dist - radii[j])
                    
                    if test_r > best_r:
                        best_r = test_r
                        best_pos = test_pos.copy()
                
                if best_r > old_r + 1e-10:
                    centers[i] = best_pos
                    radii[i] = best_r
                    improved = True
        
        if not improved:
            break
    
    # Final validation
    centers, radii = validate_and_fix(np.concatenate([centers.flatten(), radii]))
    
    return centers, radii, float(np.sum(radii))
