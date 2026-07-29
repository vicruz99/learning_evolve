# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 256846a1) state=ad151e37 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float sum of radii
    """
    n_circles = 26
    
    # --- Initialization: Hexagonal Grid ---
    # A hexagonal packing is a good starting point.
    # We try to fit rows of circles. 
    # Approximate radius for 26 circles in unit square is around 0.1.
    # We scale the grid to fit inside [0,1]x[0,1].
    
    initial_r = 0.08  # Conservative start radius
    centers = []
    
    # Generate hexagonal lattice points
    # Rows of 5 and 4 alternating might work well for 26 circles
    # 5, 4, 5, 4, 5, 3 -> Sum = 26. 
    # Let's try to distribute them evenly.
    # 26 circles. sqrt(26) ~ 5.1.
    # Maybe 6 rows? 
    # Row counts: 5, 4, 5, 4, 5, 3 (Total 26)
    row_counts = [5, 4, 5, 4, 5, 3]
    
    y_current = initial_r
    row_idx = 0
    
    for count in row_counts:
        # Horizontal spacing 2*r
        # Vertical spacing sqrt(3)*r
        # For even rows (0, 2, 4...), x starts at r.
        # For odd rows (1, 3, 5...), x starts at 2r (shifted by r) to nest.
        # Actually, standard hexagonal packing:
        # Row 0: x = r, 3r, 5r...
        # Row 1: x = 2r, 4r, 6r... (shifted by r)
        
        x_start = initial_r if row_idx % 2 == 0 else 2 * initial_r
        
        for i in range(count):
            x = x_start + i * (2 * initial_r)
            centers.append([x, y_current])
        
        y_current += initial_r * np.sqrt(3)
        row_idx += 1
        
    centers = np.array(centers)
    
    # Normalize positions to [0, 1] box roughly
    # The generated coordinates might exceed 1 or be small.
    # We scale and shift to fit roughly in [0, 1].
    # However, for optimization, starting inside is safer.
    # Let's just clamp/scale.
    # Better: Generate random valid positions if grid is messy, 
    # but hex grid is structured.
    
    # Let's rescale to fit in [0.1, 0.9] initially to give optimizer room
    min_c = centers.min()
    max_c = centers.max()
    if max_c > min_c:
        centers = (centers - min_c) / (max_c - min_c) * 0.8 + 0.1
    else:
        centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
        
    initial_radii = np.full(n_circles, initial_r)
    
    # --- Optimization Setup ---
    
    def objective(vars):
        # vars contains x0, y0, r0, x1, y1, r1, ...
        # Shape: 3 * n_circles
        radii = vars[2::3]
        return -np.sum(radii)  # Minimize negative sum

    def constraint_boundary(vars):
        # x - r >= 0  => -(x - r) <= 0
        # x + r <= 1  => x + r - 1 <= 0
        # Same for y
        cons = []
        for i in range(n_circles):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            
            cons.append(r - x)      # -(x - r)
            cons.append(x + r - 1)  # x + r - 1
            cons.append(r - y)      # -(y - r)
            cons.append(y + r - 1)  # y + r - 1
        return np.array(cons)

    def constraint_overlap(vars):
        cons = []
        for i in range(n_circles):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ri = vars[3*i+2]
            for j in range(i + 1, n_circles):
                xj = vars[3*j]
                yj = vars[3*j+1]
                rj = vars[3*j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                # Constraint: dist >= ri + rj
                # dist^2 >= (ri + rj)^2
                # (ri + rj)^2 - dist^2 <= 0
                cons.append((ri + rj)**2 - dist_sq)
        return np.array(cons)

    # Initial guess vector
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = initial_radii[i]

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Constraints definitions for scipy
    cons_boundary = {'type': 'ineq', 'fun': lambda v: -constraint_boundary(v)} # SLSQP expects >= 0
    cons_overlap = {'type': 'ineq', 'fun': lambda v: -constraint_overlap(v)}   # SLSQP expects >= 0

    # We can combine constraints or pass list
    # Note: SLSQP works better with separate constraint dicts or vectorized.
    # Vectorized form:
    def all_constraints(vars):
        c_b = -constraint_boundary(vars)
        c_o = -constraint_overlap(vars)
        return np.concatenate([c_b, c_o])

    # SLSQP requires Jacobian for large problems, but numerical approx is okay for 78 vars.
    # However, it might be slow. Let's try.
    
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints={'type': 'ineq', 'fun': all_constraints},
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if result.success:
        final_vars = result.x
    else:
        # Fallback or retry logic could go here, but we proceed with best found
        final_vars = x0

    # Extract results
    final_centers = np.array([[final_vars[3*i], final_vars[3*i+1]] for i in range(n_circles)])
    final_radii = np.array([final_vars[3*i+2] for i in range(n_circles)])
    sum_radii = np.sum(final_radii)

    # Sort radii for consistency (optional, but good for debugging)
    # But we must keep centers aligned.
    # The order doesn't matter for validation.
    
    return final_centers, final_radii, sum_radii

if __name__ == "__main__":
    # Basic check
    import numpy as np
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any():
            return False
        for i in range(n):
            if radii[i] < 0: return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")
