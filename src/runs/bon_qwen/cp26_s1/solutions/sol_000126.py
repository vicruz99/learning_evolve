# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22de7e34) state=d0bf1197 sum of radii=2.613222 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def evaluate_solution(centers, radii):
    n = len(radii)
    total_sum = np.sum(radii)
    valid = True
    
    # Check bounds
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            valid = False
            
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_distance(centers[i], centers[j])
            if dist < radii[i] + radii[j] - 1e-12:
                valid = False
                
    return valid, total_sum

def get_hexagonal_centers(n_circles):
    centers = []
    r_guess = 0.1 
    y = r_guess
    row = 0
    while len(centers) < n_circles:
        shift = r_guess if row % 2 == 1 else 0
        x = r_guess + shift
        while x <= 1 - r_guess:
            centers.append((x, y))
            x += 2 * r_guess
            if len(centers) >= n_circles:
                break
        y += math.sqrt(3) * r_guess
        row += 1
    return np.array(centers[:n_circles])

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Optimization function wrapper
    def objective(x):
        # x contains [x1, y1, r1, x2, y2, r2, ...]
        radii = x[2::3]
        return -np.sum(radii)

    def constraint_overlap(x, i, j):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
        dist_sq = (xi - xj)**2 + (yi - yj)**2
        sum_r = ri + rj
        return dist_sq - sum_r**2

    def constraint_wall_left(x, i):
        return x[3*i] - x[3*i+2]

    def constraint_wall_right(x, i):
        return (1 - x[3*i]) - x[3*i+2]

    def constraint_wall_bottom(x, i):
        return x[3*i+1] - x[3*i+2]

    def constraint_wall_top(x, i):
        return (1 - x[3*i+1]) - x[3*i+2]

    def constraint_radius_pos(x, i):
        return x[3*i+2]

    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: constraint_overlap(x, i, j)})
        
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_wall_left(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_wall_right(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_wall_bottom(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_wall_top(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_radius_pos(x, i)})

    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal grid
    base_centers = get_hexagonal_centers(n)
    initial_radii = np.full(n, 0.08)
    x0_1 = []
    for i in range(n):
        x0_1.extend([base_centers[i, 0], base_centers[i, 1], initial_radii[i]])
    initial_configs.append(x0_1)

    # Config 2: Random perturbation
    for _ in range(3):
        centers_pert = base_centers.copy()
        centers_pert += np.random.uniform(-0.02, 0.02, centers_pert.shape)
        centers_pert = np.clip(centers_pert, 0.1, 0.9)
        x0_temp = []
        for i in range(n):
            x0_temp.extend([centers_pert[i, 0], centers_pert[i, 1], 0.08])
        initial_configs.append(x0_temp)
        
    # Config 3: Rotated grid
    angle = np.pi / 12 
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, -s], [s, c]])
    centers_rot = (base_centers - 0.5) @ rot_matrix.T + 0.5
    centers_rot = np.clip(centers_rot, 0.1, 0.9)
    x0_3 = []
    for i in range(n):
        x0_3.extend([centers_rot[i, 0], centers_rot[i, 1], 0.08])
    initial_configs.append(x0_3)

    # Run optimization
    for x0 in initial_configs:
        try:
            res = scipy.optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success:
                radii_opt = res.x[2::3]
                centers_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                current_sum = np.sum(radii_opt)
                
                valid, _ = evaluate_solution(centers_opt, radii_opt)
                if valid and current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers_opt
                    best_radii = radii_opt
        except Exception:
            continue

    # Final safety check and fallback to simple grid if optimization fails
    if best_centers is None:
        centers_fallback = get_hexagonal_centers(n)
        radii_fallback = np.full(n, 0.08)
        best_centers = centers_fallback
        best_radii = radii_fallback
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
