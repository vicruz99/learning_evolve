# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc549794) state=b0ff3e08 sum of radii=0.443357 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)

    # 1. Initialize centers with a hexagonal lattice pattern
    centers = np.zeros((n, 2))
    row_count = 0
    # Create a rough hexagonal grid
    spacing = 0.25
    for r in range(10):
        for c in range(10):
            if len(centers) == n:
                break
            x = c * spacing + (r % 2) * (spacing / 2)
            y = r * spacing * np.sqrt(3) / 2
            # Fit into unit square with some padding
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers[len(centers)] = [x, y]
        if len(centers) == n:
            break
    
    # If we didn't fill it (unlikely), fill randomly
    while len(centers) < n:
        centers[len(centers)] = np.random.rand(2)

    # 2. Define the LP solver function to find max radii for fixed centers
    def solve_radii(centers):
        n = centers.shape[0]
        # Variables: r_0, ..., r_{n-1}
        c_obj = -np.ones(n) # Maximize sum(r) -> Minimize -sum(r)
        
        A_ub = []
        b_ub = []
        
        # Pairwise distance constraints: r_i + r_j <= dist(i, j)
        # dist(i, j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
        
        # Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            
            A_ub.append(row); b_ub.append(centers[i, 0])
            A_ub.append(row); b_ub.append(1.0 - centers[i, 0])
            A_ub.append(row); b_ub.append(centers[i, 1])
            A_ub.append(row); b_ub.append(1.0 - centers[i, 1])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for r_i >= 0
        bounds = [(0, None) for _ in range(n)]
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            radii = res.x
            return radii, -res.fun
        else:
            # Fallback to small valid radii if LP fails
            return np.ones(n) * 0.01, 0.26

    # 3. Hill Climbing Optimization
    current_sum, best_sum = 0.0, 0.0
    current_radii, _ = solve_radii(centers)
    best_radii = current_radii.copy()
    
    # Calculate initial sum
    _, best_sum = solve_radii(centers)
    
    # Optimization parameters
    steps = 2000
    step_size = 0.02
    
    for _ in range(steps):
        # Pick a random circle to move
        idx = np.random.randint(n)
        old_pos = centers[idx].copy()
        
        # Perturb position
        move_x = np.random.uniform(-step_size, step_size)
        move_y = np.random.uniform(-step_size, step_size)
        
        centers[idx, 0] = np.clip(old_pos[0] + move_x, 0, 1)
        centers[idx, 1] = np.clip(old_pos[1] + move_y, 0, 1)
        
        # Solve for new radii
        current_radii, current_sum = solve_radii(centers)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_radii = current_radii.copy()
        else:
            # Revert move
            centers[idx] = old_pos

    # 4. Final Validation Check
    if validate_packing(centers, best_radii):
        return centers, best_radii, best_sum
    else:
        # Fallback to equal radii if validation fails (shouldn't happen)
        r_fallback = 0.05
        centers_fallback = np.random.rand(n, 2) * 0.9 + 0.05
        return centers_fallback, np.ones(n) * r_fallback, n * r_fallback

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

# Run the packing function to generate the solution
if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    print(f"Validation: {validate_packing(centers, radii)}")
