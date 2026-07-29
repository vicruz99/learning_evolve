# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0cda7bbd) state=cd1c4815 sum of radii=2.611359 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    def get_constraints(centers, radii):
        """Generates constraint values for scipy optimization."""
        cons = []
        
        # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # Converted to form g(x) >= 0
        for i in range(n):
            # x >= r  => x - r >= 0
            cons.append(centers[i, 0] - radii[i])
            # 1 - x >= r => 1 - x - r >= 0
            cons.append(1 - centers[i, 0] - radii[i])
            # y >= r  => y - r >= 0
            cons.append(centers[i, 1] - radii[i])
            # 1 - y >= r => 1 - y - r >= 0
            cons.append(1 - centers[i, 1] - radii[i])
            
        # Non-overlap constraints: dist >= r1 + r2
        # (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
        # To make it differentiable and smoother for optimizer, we can use linearized constraints 
        # or keep quadratic. SLSQP handles quadratic constraints.
        # Constraint: (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                cons.append(dist_sq - r_sum*r_sum)
                
        return cons

    def objective(vars):
        """Objective to minimize: -sum(radii)"""
        r = vars[2::3]
        return -np.sum(r)

    def vars_to_state(vars):
        """Convert flat vector to centers and radii arrays"""
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        centers[:, 0] = vars[0::3]
        centers[:, 1] = vars[1::3]
        radii = vars[2::3]
        return centers, radii

    # Define bounds for variables [x, y, r]
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])

    best_solution = None
    best_val = -1e9

    # Helper to run optimization for a given initial state
    def try_optimize(init_centers, init_radii):
        nonlocal best_solution, best_val
        
        init_vars = np.zeros(3 * n)
        init_vars[0::3] = init_centers[:, 0]
        init_vars[1::3] = init_centers[:, 1]
        init_vars[2::3] = init_radii
        
        # Constraints callback
        cons = {'type': 'ineq', 'fun': lambda v: get_constraints(vars_to_state(v)[0], vars_to_state(v)[1])}
        
        try:
            res = opt.minimize(objective, init_vars, method='SLSQP', bounds=bounds, constraints=cons, 
                               options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success:
                centers, radii = vars_to_state(res.x)
                # Post-validation to ensure strict feasibility due to numerical errors
                # Apply a small shrinkage to radii to ensure strict inequality if close
                # But better to just check and repair if needed
                
                # Calculate sum
                s = np.sum(radii)
                if s > best_val:
                    # Verify validity manually before updating best
                    if validate_packing(centers, radii):
                        best_val = s
                        best_solution = (centers.copy(), radii.copy())
        except Exception:
            pass

    # --- Initial Configurations ---

    # 1. Grid 5x5 + 1 small circle in gap
    c1 = []
    r1 = []
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            c1.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
            r1.append(0.1)
    # 26th circle in gap at (0.2, 0.2) relative to grid? 
    # Grid centers: 0.1, 0.3, 0.5, 0.7, 0.9
    # Gap center between (0.1,0.1) and (0.3,0.1) is (0.2, 0.1) - no, that's edge.
    # Gap between (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3) is (0.2, 0.2).
    c1.append([0.2, 0.2])
    r1.append(0.04) # Safe radius
    init_c1 = np.array(c1)
    init_r1 = np.array(r1)
    try_optimize(init_c1, init_r1)

    # 2. Hexagonal packing approximation
    # 5 rows. Approx radii 0.095
    c2 = []
    r2 = []
    r_hex = 0.095
    y_step = np.sqrt(3) * r_hex
    # 5 rows
    rows = [5, 6, 5, 6, 4] # Sum 26? 5+6+5+6+4 = 26.
    # Actually 5+6+5+6+4 = 26.
    # Let's try to fit them.
    # Row 0: 5 circles. y = r_hex. x starts at r_hex.
    # Row 1: 6 circles. y = r_hex + y_step. x starts at 2*r_hex (offset).
    # ...
    y_curr = r_hex
    for i, count in enumerate(rows):
        x_start = r_hex if i % 2 == 0 else 2 * r_hex # Offset by r
        # Actually standard hex offset is r.
        # If row 0 starts at r, row 1 starts at 2r? No, center to center horizontal dist is 2r.
        # Offset means centers are shifted by r horizontally.
        # If row 0 centers at r, 3r, 5r...
        # Row 1 centers at 2r, 4r, 6r...
        
        for k in range(count):
            x = x_start + k * 2 * r_hex
            # Adjust if x goes out of bounds?
            # If x > 1-r, we might need to scale or shift.
            # Let's just place them and let optimizer fix.
            c2.append([x, y_curr])
            r2.append(r_hex)
        y_curr += y_step
    
    init_c2 = np.array(c2)
    init_r2 = np.array(r2)
    try_optimize(init_c2, init_r2)

    # 3. Random perturbation of Grid
    c3 = init_c1.copy()
    r3 = init_r1.copy()
    # Add small noise
    np.random.seed(42)
    noise_c = np.random.uniform(-0.01, 0.01, size=c3.shape)
    noise_r = np.random.uniform(-0.005, 0.005, size=r3.shape)
    # Ensure positive radii
    noise_r = np.abs(noise_r)
    
    c3 = c3 + noise_c
    r3 = r3 + noise_r
    
    # Clip to bounds
    c3[:, 0] = np.clip(c3[:, 0], 0.05, 0.95)
    c3[:, 1] = np.clip(c3[:, 1], 0.05, 0.95)
    r3 = np.clip(r3, 0.01, 0.2)
    
    try_optimize(c3, r3)

    # 4. Another random start with smaller radii to allow more freedom
    c4 = np.random.uniform(0.1, 0.9, size=(n, 2))
    r4 = np.full(n, 0.05)
    try_optimize(c4, r4)

    # If no solution found (should not happen with grid start), return grid
    if best_solution is None:
        return init_c1, init_r1, np.sum(init_r1)

    centers, radii = best_solution
    return centers, radii, np.sum(radii)

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        if np.isnan(radii[i]):
            return False

    for i in range(n):
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
