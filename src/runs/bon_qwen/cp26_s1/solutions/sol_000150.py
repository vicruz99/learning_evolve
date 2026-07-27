# sol_000150 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 08bd70cf) state=d7300d79 sum of radii=0.608873 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_radii_lp(centers):
    """
    Given fixed centers, solve for the optimal radii that maximize sum(r)
    using Linear Programming.
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(x)
        # r_i <= 1-x
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(1.0 - x)
        # r_i <= y
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(y)
        # r_i <= 1-y
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(1.0 - y)
        
    # 2. Pairwise non-overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n); row[i] = 1.0; row[j] = 1.0
            A_ub.append(row); b_ub.append(dist)
            
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback to equal small radii if LP fails (should not happen)
        return np.full(n, 0.01)

def optimize_centers(n_circles=26, iterations=5000):
    """
    Uses a force-directed repulsion approach to find a good set of centers.
    """
    # Initialize in a dense hexagonal-like grid
    centers = np.zeros((n_circles, 2))
    idx = 0
    # 5 rows pattern
    row_counts = [5, 6, 5, 6, 4] # Total 26
    y_step = 1.0 / 6.0
    
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * (1.0 / 5.0) # Center rows vertically
        x_step = 1.0 / (count + 1)
        for j in range(count):
            x = (j + 1) * x_step
            # Shift even rows for hexagonal packing
            if i % 2 == 1:
                x += x_step / 2
            centers[idx] = [x, y]
            idx += 1
            
    # Ensure indices are correct and within bounds
    np.random.shuffle(centers) # Add slight randomness to avoid symmetry
    centers = np.clip(centers, 0.1, 0.9)
    
    # Force-directed optimization
    # Repulsive force F = k / d^2
    k = 0.1
    dt = 0.05
    friction = 0.9
    
    for _ in range(iterations):
        forces = np.zeros_like(centers)
        
        # Calculate pairwise repulsion
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < 1e-10:
                    continue
                dist = np.sqrt(dist_sq)
                # Repulsive force magnitude
                f_mag = k / dist_sq
                f_vec = f_mag * diff / dist
                forces[i] += f_vec
                forces[j] -= f_vec
                
        # Wall repulsion (keep inside [0,1])
        for i in range(n_circles):
            if centers[i, 0] < 0.05: forces[i, 0] += k * 10
            if centers[i, 0] > 0.95: forces[i, 0] -= k * 10
            if centers[i, 1] < 0.05: forces[i, 1] += k * 10
            if centers[i, 1] > 0.95: forces[i, 1] -= k * 10

        # Update positions
        centers += dt * forces
        # Apply friction/damping to settle
        # (Not strictly necessary for simple gradient step, but helps stability)
        
        # Project back to valid box [0,1]
        centers = np.clip(centers, 0, 1)
        
        # Reduce step size over time
        dt *= 0.9995

    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # 1. Find optimized centers using force-directed repulsion
    # We run a quick optimization to spread points out
    centers = optimize_centers(n_circles=n, iterations=2000)
    
    # 2. Solve for optimal radii using Linear Programming
    radii = solve_radii_lp(centers)
    
    # 3. Calculate sum of radii
    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)

# Final block to ensure function is callable and returns correct types
if __name__ == "__main__":
    centers, radii, total_radius = run_packing()
    print(f"Sum of radii: {total_radius}")
