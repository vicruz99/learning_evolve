# sol_000117 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4a327247) state=c96a11b2 sum of radii=2.498307 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """
    Solves for optimal radii given fixed centers using Linear Programming.
    Maximizes sum(radii) subject to non-overlap and boundary constraints.
    
    Args:
        centers: np.array of shape (n, 2)
        
    Returns:
        radii: np.array of shape (n)
        marginals: np.array of shadow prices (dual variables) or None
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(radii) => Minimize -sum(radii)
    c = -np.ones(n)
    
    # Bounds for radii: 0 <= r_i <= distance to nearest wall
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r_wall = min(x, 1-x, y, 1-y)
        max_r_wall = max(0.0, max_r_wall)
        bounds.append((0, max_r_wall))
        
    # Inequality constraints: r_i + r_j <= dist(i, j)
    num_pairs = n * (n - 1) // 2
    if num_pairs > 0:
        A_ub = np.zeros((num_pairs, n))
        b_ub = np.zeros(num_pairs)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                A_ub[idx, i] = 1
                A_ub[idx, j] = 1
                b_ub[idx] = dist
                idx += 1
    else:
        A_ub = None
        b_ub = None

    try:
        # Use 'highs' method for speed
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            radii = res.x
            marginals = None
            # Retrieve dual variables (shadow prices) if available
            # These indicate how much the objective would improve if the constraint was relaxed
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                marginals = res.ineqlin.marginals
            return radii, marginals
        else:
            return np.zeros(n), None
    except Exception:
        return np.zeros(n), None

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a combination of Linear Programming for radius optimization 
    and force-directed layout for center positioning.
    """
    n = 26
    
    # --- Initialization ---
    # Start with a grid pattern perturbed to break symmetry
    centers = np.zeros((n, 2))
    
    pts = []
    # 5x5 grid covers 25 circles
    for r in range(5):
        for c in range(5):
            x = (c + 0.5) / 5.0
            y = (r + 0.5) / 5.0
            pts.append([x, y])
    
    # Add 26th circle in a gap. 
    # Grid points are at 0.1, 0.3, 0.5, 0.7, 0.9.
    # Gaps are centered at 0.2, 0.4, 0.6, 0.8.
    # (0.2, 0.2) is a good candidate.
    pts.append([0.2, 0.2])
    
    centers = np.array(pts[:26])
    
    # Perturb to avoid perfect symmetry which might hinder gradient based methods
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # --- Optimization ---
    best_sum = -1.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    step_size = 0.02
    decay = 0.96
    
    # Optimization Loop
    for it in range(150):
        radii, duals = get_optimal_radii(centers)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute forces based on duals (shadow prices)
        # If a constraint is active (mu > 0), moving centers apart helps.
        forces = np.zeros_like(centers)
        
        if duals is not None:
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    mu = duals[idx]
                    if mu > 1e-6: # Active constraint
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            # Force vector proportional to gradient of distance
                            # Direction away from each other
                            fx = mu * dx / dist
                            fy = mu * dy / dist
                            forces[i] += [fx, fy]
                            forces[j] -= [fx, fy]
                    idx += 1
        
        # Wall forces: push away if radius is limited by wall
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Check if radius is limited by wall (approximate check)
            # If r is close to dist to wall, force away
            if r > 1e-6:
                dist_left = x
                dist_right = 1 - x
                dist_bottom = y
                dist_top = 1 - y
                
                # Threshold for "touching" wall
                if abs(dist_left - r) < 1e-4:
                    forces[i, 0] += 1.0
                if abs(dist_right - r) < 1e-4:
                    forces[i, 0] -= 1.0
                if abs(dist_bottom - r) < 1e-4:
                    forces[i, 1] += 1.0
                if abs(dist_top - r) < 1e-4:
                    forces[i, 1] -= 1.0
                
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        forces_norm = forces / norms
        
        centers += step_size * forces_norm
        
        # Keep centers within valid bounds (must allow r > 0)
        centers = np.clip(centers, 0.01, 0.99)
        
        # Decay step size
        step_size *= decay
        
        # Random perturbation to escape local optima
        if it % 30 == 0:
            centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
            centers = np.clip(centers, 0.01, 0.99)

    # Final validation step to ensure consistency
    final_radii, _ = get_optimal_radii(best_centers)
    final_radii = np.maximum(final_radii, 0)
    
    return best_centers, final_radii, float(np.sum(final_radii))
