# sol_000127 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 873af2c6) state=7a4844ab sum of radii=2.529454 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

N_CIRCLES = 26
TOLERANCE = 1e-4
FORCE_STRENGTH = 1.0
BOUNDARY_FORCE = 1.5
STEP_DECAY = 0.99

def get_constraints_matrix(centers):
    """Constructs the inequality matrix for the LP solver."""
    n = centers.shape[0]
    n_boundary = 4 * n
    n_pairs = n * (n - 1) // 2
    total_constraints = n_boundary + n_pairs

    A_ub = np.zeros((total_constraints, n))
    b_ub = np.zeros(total_constraints)

    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        row = 4 * i
        A_ub[row, i] = 1.0; b_ub[row] = x
        row += 1
        A_ub[row, i] = 1.0; b_ub[row] = 1.0 - x
        row += 1
        A_ub[row, i] = 1.0; b_ub[row] = y
        row += 1
        A_ub[row, i] = 1.0; b_ub[row] = 1.0 - y

    # Pairwise constraints: r_i + r_j <= dist(i, j)
    row = n_boundary
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < 1e-9:
                dist = 1e-9
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = dist
            row += 1

    return A_ub, b_ub

def solve_for_radii(centers):
    """Solves the LP to find max sum of radii for fixed centers."""
    n = centers.shape[0]
    A_ub, b_ub = get_constraints_matrix(centers)
    c = np.ones(n) * -1.0
    bounds = [(0, None) for _ in range(n)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, res.fun * -1
    return np.full(n, 0.01), 0.0

def compute_forces(centers, radii):
    """Calculates repulsive forces to unlock jammed circles."""
    n = centers.shape[0]
    forces = np.zeros_like(centers)

    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < 1e-9: continue
            
            r_sum = radii[i] + radii[j]
            if dist - r_sum < TOLERANCE * 2:
                vec = centers[i] - centers[j]
                dir_vec = vec / dist
                force_mag = FORCE_STRENGTH * (2.0 - (dist - r_sum) / TOLERANCE)
                forces[i] += force_mag * dir_vec
                forces[j] -= force_mag * dir_vec

    # Boundary repulsion
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < TOLERANCE: forces[i, 0] += BOUNDARY_FORCE
        if 1.0 - x - r < TOLERANCE: forces[i, 0] -= BOUNDARY_FORCE
        if y - r < TOLERANCE: forces[i, 1] += BOUNDARY_FORCE
        if 1.0 - y - r < TOLERANCE: forces[i, 1] -= BOUNDARY_FORCE

    return forces

def run_packing():
    centers = np.zeros((N_CIRCLES, 2))
    idx = 0
    
    # Hexagonal grid initialization
    row_y = 0.1
    while idx < N_CIRCLES and row_y <= 0.9:
        is_odd = int((row_y - 0.1) // 0.2) % 2
        col_x = 0.1
        if is_odd: col_x = 0.15
        
        while col_x <= 0.9 and idx < N_CIRCLES:
            centers[idx, 0] = col_x
            centers[idx, 1] = row_y
            col_x += 0.25
            idx += 1
        row_y += 0.25
        
    # Fallback fill if grid didn't reach 26
    while idx < N_CIRCLES:
        centers[idx, 0] = np.random.rand() * 0.8 + 0.1
        centers[idx, 1] = np.random.rand() * 0.8 + 0.1
        idx += 1

    step_size = 0.05
    radii, _ = solve_for_radii(centers)

    # Optimization loop
    for iteration in range(300):
        radii, current_sum = solve_for_radii(centers)
        forces = compute_forces(centers, radii)
        centers += step_size * forces
        
        # Clamp centers to [0, 1]
        centers = np.clip(centers, 0.001, 0.999)
        step_size *= STEP_DECAY

    # Final high-precision solve
    radii, sum_radii = solve_for_radii(centers)
    return centers, radii, sum_radii
