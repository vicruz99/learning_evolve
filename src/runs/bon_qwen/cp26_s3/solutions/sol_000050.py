# sol_000050 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state baeb2167) state=d0ea1493 sum of radii=2.076796 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def get_uv(centers, radii):
    """
    Convert centers and radii to parameterization (u, v).
    x = r + (1 - 2r) * u
    y = r + (1 - 2r) * v
    This ensures x, y stay within [r, 1-r] if u, v in [0, 1].
    """
    n = len(radii)
    u = np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        denom = 1 - 2 * radii[i]
        if denom < 1e-9:
            # If r is close to 0.5, center must be at 0.5
            u[i] = 0.5
            v[i] = 0.5
        else:
            u[i] = (centers[i, 0] - radii[i]) / denom
            v[i] = (centers[i, 1] - radii[i]) / denom
    return u, v

def get_centers(radii, u, v):
    """
    Convert radii and parameters (u, v) back to centers.
    """
    n = len(radii)
    centers = np.zeros((n, 2))
    for i in range(n):
        denom = 1 - 2 * radii[i]
        centers[i, 0] = radii[i] + denom * u[i]
        centers[i, 1] = radii[i] + denom * v[i]
    return centers

def objective(vars, n_circles):
    """
    Objective function for optimization.
    Maximizes sum of radii (minimizes -sum(r)) with penalty for overlaps.
    """
    r = vars[:n_circles]
    u = vars[n_circles:2*n_circles]
    v = vars[2*n_circles:3*n_circles]
    
    centers = get_centers(r, u, v)
    
    penalty = 0.0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            c1 = centers[i]
            c2 = centers[j]
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Avoid singularity at dist=0
            if dist < 1e-9:
                dist = 1e-9
            
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += overlap**2
    
    # We want to maximize sum(r), so we minimize -sum(r) + penalty
    return -np.sum(r) + 10000.0 * penalty

def run_packing():
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    n_circles = N_CIRCLES
    
    # 1. Initialization
    # Start with a 5x5 grid (25 circles) and add 1 circle in a gap.
    # Grid points at 0.1, 0.3, 0.5, 0.7, 0.9.
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    grid_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    grid_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    idx = 0
    for y in grid_y:
        for x in grid_x:
            centers[idx] = [x, y]
            idx += 1
    
    # Place the 26th circle in a gap (e.g., center of the hole between (0.1,0.1), (0.3,0.1), etc.)
    centers[25] = [0.2, 0.2]
    
    # Initial radius small enough to avoid overlaps (dist to neighbor ~0.14, so 0.05 is safe)
    r_init = 0.05 
    radii[:] = r_init
    
    # Add small noise to break symmetry and help escape local minima
    np.random.seed(42)
    centers += np.random.uniform(-0.001, 0.001, centers.shape)
    # Ensure centers are valid for initial radii
    centers = np.clip(centers, r_init, 1-r_init)
    
    # 2. Force-based Expansion (Jiggling)
    # Iteratively grow radii and resolve overlaps to find a dense packing configuration.
    n_steps = 500
    dt = 0.01
    k_rep = 20.0   # Repulsion strength between circles
    k_wall = 50.0  # Repulsion strength from walls
    
    for step in range(n_steps):
        # Slowly grow radii
        radii *= 1.0001 
        
        forces = np.zeros_like(centers)
        
        # Inter-circle repulsion
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff**2)
                
                if dist_sq < 1e-12:
                    dist = 1e-6
                    diff = np.array([1e-6, 0.0]) # Arbitrary direction
                else:
                    dist = np.sqrt(dist_sq)
                
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    f = k_rep * overlap
                    fx = f * diff[0] / dist
                    fy = f * diff[1] / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall repulsion
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += k_wall * (r - x)
            # Right wall
            elif x > 1 - r:
                forces[i, 0] -= k_wall * (x - (1 - r))
            
            # Bottom wall
            if y < r:
                forces[i, 1] += k_wall * (r - y)
            # Top wall
            elif y > 1 - r:
                forces[i, 1] -= k_wall * (y - (1 - r))
        
        # Update centers
        centers += dt * forces
        # Clamp to [0, 1] to prevent flying out (though wall forces should handle it)
        centers = np.clip(centers, 0, 1)

    # 3. Optimization
    # Map current state to (r, u, v) parameters for boundary-safe optimization
    u_curr, v_curr = get_uv(centers, radii)
    
    # Clip values to valid ranges
    radii = np.clip(radii, 0, 0.49)
    u_curr = np.clip(u_curr, 0, 1)
    v_curr = np.clip(v_curr, 0, 1)
    
    # Flatten to 1D vector: [r1..r26, u1..u26, v1..v26]
    x0 = np.concatenate([radii, u_curr, v_curr])
    
    # Bounds: r in [0, 0.5], u in [0, 1], v in [0, 1]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        
    try:
        # Use L-BFGS-B for bounded optimization
        res = minimize(objective, x0, args=(n_circles,), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12})
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    # Extract optimized variables
    r_opt = x_opt[:n_circles]
    u_opt = x_opt[n_circles:2*n_circles]
    v_opt = x_opt[2*n_circles:3*n_circles]
    
    # Reconstruct centers
    centers_opt = get_centers(r_opt, u_opt, v_opt)
    
    # Ensure radii are non-negative
    r_opt = np.maximum(r_opt, 0)
    centers_opt = get_centers(r_opt, u_opt, v_opt)
    
    # Post-processing: shrink radii if any overlaps remain (safety check)
    for _ in range(50):
        max_overlap = 0
        pair = (-1, -1)
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                sum_r = r_opt[i] + r_opt[j]
                if dist < sum_r:
                    overlap = sum_r - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                        pair = (i, j)
        
        if max_overlap > 1e-9:
            # Reduce radii of overlapping pair
            reduction = max_overlap / 2
            r_opt[pair[0]] = max(0, r_opt[pair[0]] - reduction)
            r_opt[pair[1]] = max(0, r_opt[pair[1]] - reduction)
            centers_opt = get_centers(r_opt, u_opt, v_opt)
        else:
            break
            
    sum_radii = float(np.sum(r_opt))
    
    return centers_opt, r_opt, sum_radii
