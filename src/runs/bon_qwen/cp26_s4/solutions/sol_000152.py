# sol_000152 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3a06727e) state=51d1c2c9 sum of radii=2.516663 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(params, n, mu):
    """
    Computes the objective function: -sum(radii) + penalties for violations.
    
    Args:
        params: np.array of shape (3*n,), flat vector [x0, y0, r0, x1, y1, r1, ...]
        n: Number of circles
        mu: Penalty weight
    
    Returns:
        Scalar objective value
    """
    # Reshape parameters
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Main objective: maximize sum of radii -> minimize -sum
    obj = -np.sum(rs)
    
    # 1. Pairwise Overlap Penalty
    # Calculate all pairwise distances efficiently
    # dx[i, j] = xs[i] - xs[j]
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Sum of radii matrix
    sum_radii = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    # Overlap amount: positive means overlap
    overlap = sum_radii - dist
    
    # We only consider pairs i < j to avoid double counting and self-interaction
    # Create mask for upper triangle excluding diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise_overlap = overlap[mask]
    
    # Penalize positive overlaps (squared penalty)
    penalty_overlap = np.maximum(0.0, pairwise_overlap)**2
    obj += mu * np.sum(penalty_overlap)
    
    # 2. Boundary Penalties
    # Constraints:
    # x >= r  => r - x <= 0  => violation if r - x > 0
    # x <= 1-r => x + r - 1 <= 0 => violation if x + r - 1 > 0
    # Same for y
    
    # x < r violation
    val_x_low = rs - xs
    obj += mu * np.sum(np.maximum(0.0, val_x_low)**2)
    
    # x > 1 - r violation
    val_x_high = xs + rs - 1.0
    obj += mu * np.sum(np.maximum(0.0, val_x_high)**2)
    
    # y < r violation
    val_y_low = rs - ys
    obj += mu * np.sum(np.maximum(0.0, val_y_low)**2)
    
    # y > 1 - r violation
    val_y_high = ys + rs - 1.0
    obj += mu * np.sum(np.maximum(0.0, val_y_high)**2)
    
    # 3. Non-negative radius penalty (should be covered by bounds, but safe to have)
    # r >= 0 => violation if r < 0
    val_r_neg = -rs
    obj += mu * np.sum(np.maximum(0.0, val_r_neg)**2)
    
    return obj

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- Initialization ---
    # Strategy: 5x5 grid for 25 circles, plus 1 circle in a gap.
    # Grid centers at 0.1, 0.3, 0.5, 0.7, 0.9
    # This grid has spacing 0.2. If r=0.1, they touch.
    # We start with smaller r to allow movement.
    
    centers_init = []
    
    # Add 25 circles in 5x5 grid
    for i in range(5):
        for j in range(5):
            x = 0.1 + 0.2 * i
            y = 0.1 + 0.2 * j
            centers_init.append([x, y])
    
    # Add 26th circle in a gap. 
    # A gap exists at (0.2, 0.2) relative to grid points (0.1, 0.1), (0.3, 0.1), etc.
    # Distance to neighbors is sqrt(0.1^2 + 0.1^2) = 0.1414.
    # With r=0.05, sum r = 0.1 < 0.1414, so no overlap initially.
    centers_init.append([0.2, 0.2])
    
    centers_init = np.array(centers_init)
    radii_init = np.full(n, 0.05) # Start small
    
    # Flatten parameters: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # --- Optimization ---
    # We use a penalty method. 
    # Start with a moderate mu and optimize.
    # A very large mu might make the landscape too steep for L-BFGS-B if starting far from solution.
    # But our start is valid (no overlap), so gradient of penalty is 0 initially.
    # Optimizer will increase radii until constraints bind.
    
    mu = 5000.0 
    
    # Run optimization
    # maxiter increased to allow thorough search
    res = minimize(
        compute_objective, 
        x0, 
        args=(n, mu), 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    # Extract results
    final_params = res.x
    final_xs = final_params[0::3]
    final_ys = final_params[1::3]
    final_rs = final_params[2::3]
    
    centers = np.column_stack((final_xs, final_ys))
    sum_radii = np.sum(final_rs)
    
    # Post-processing: Ensure strict validity for the validator
    # The optimizer might leave tiny overlaps due to numerical tolerance.
    # We can try to fix by slightly reducing radii if needed, but 
    # the penalty method with high mu usually keeps them feasible or very close.
    # However, to be safe, we can run a check and shrink if necessary.
    
    # Let's verify validity internally and adjust if needed
    # (This is just a safety net, not strictly required if optimizer is good)
    
    # Check overlaps
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < final_rs[i] + final_rs[j] - 1e-12:
                valid = False
                break
        if not valid: break
    
    # Check boundaries
    for i in range(n):
        if (centers[i, 0] - final_rs[i] < -1e-12 or 
            centers[i, 0] + final_rs[i] > 1 + 1e-12 or
            centers[i, 1] - final_rs[i] < -1e-12 or 
            centers[i, 1] + final_rs[i] > 1 + 1e-12):
            valid = False
            break
            
    if not valid:
        # If invalid, shrink radii slightly to make it valid
        # This is a fallback
        factor = 0.99
        final_rs *= factor
        # Re-center if out of bounds (unlikely if only radius shrank)
        # But radius shrink fixes boundary issues too.
        # Overlaps: reducing radii helps.
        # We might need to iterate shrinking, but 0.99 should be enough for small violations.
        sum_radii = np.sum(final_rs)

    return centers, final_rs, sum_radii
