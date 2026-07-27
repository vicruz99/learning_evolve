import numpy as np
import scipy.optimize as opt
import math

def get_boundary_distances(centers):
    """Calculate distance from each center to the nearest boundary."""
    x = centers[:, 0]
    y = centers[:, 1]
    dists = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    return dists

def compute_pairwise_distances(centers):
    """Compute pairwise distances between centers."""
    n = centers.shape[0]
    # Vectorized computation of squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dist_sq)
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    return dists

def solve_radii_lp(centers):
    """
    Solve LP to maximize sum of radii given fixed centers.
    Constraints:
    r_i + r_j <= dist_ij
    r_i <= dist_to_boundary_i
    r_i >= 0
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Bounds for radii: r_i >= 0
    # We will handle upper bounds via inequality constraints
    
    # Inequality constraints A_ub @ x <= b_ub
    # 1. Pairwise: r_i + r_j <= dist_ij
    # 2. Boundary: r_i <= boundary_dist_i
    
    n_constraints = 0
    A_ub = []
    b_ub = []
    
    # Precompute distances
    dists = compute_pairwise_distances(centers)
    b_dists = get_boundary_distances(centers)
    
    # Add boundary constraints: r_i <= b_dists[i]
    # This is -r_i >= -b_dists[i] -> r_i <= b_dists[i]
    # Actually standard form is A x <= b.
    # So 1*r_i <= b_dists[i]
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(b_dists[i])
        
    # Add pairwise constraints: r_i + r_j <= dists[i, j]
    # Only need upper triangle i < j
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for variables: (lower, upper)
    # r_i >= 0. Upper bound can be loose (e.g., 1.0) or derived from boundary.
    # linprog handles bounds.
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: return small radii
            return np.full(n, 1e-6)
    except Exception:
        return np.full(n, 1e-6)

def relax_centers(centers, radii, iterations=100):
    """
    Move centers to reduce overlaps and increase space, using a force-directed approach.
    Radii are treated as target sizes to maintain or improve upon.
    """
    n = centers.shape[0]
    best_centers = centers.copy()
    best_score = -np.sum(radii) # We want to maximize sum, so minimize negative sum? 
                                # Actually, this function just tries to fix overlaps.
                                # Let's just push apart if dist < r_i + r_j.
    
    # Current radii
    r = radii.copy()
    
    # Learning rate for movement
    alpha = 0.05
    
    for _ in range(iterations):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[j] - centers[i]
                dist = np.linalg.norm(vec)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Overlap or too close
                    # Force proportional to overlap
                    if dist > 1e-9:
                        overlap = min_dist - dist
                        force_mag = overlap * 0.5 # Damping
                        force_dir = vec / dist
                        forces[i] -= force_dir * force_mag
                        forces[j] += force_dir * force_mag
                    else:
                        # Very close, push random direction
                        rand_dir = np.random.randn(2)
                        forces[i] -= rand_dir * 0.01
                        forces[j] += rand_dir * 0.01
        
        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            rad = r[i]
            
            # Left
            if x - rad < 0:
                forces[i, 0] += (rad - x) * 0.5
            # Right
            if x + rad > 1:
                forces[i, 0] -= (x + rad - 1) * 0.5
            # Bottom
            if y - rad < 0:
                forces[i, 1] += (rad - y) * 0.5
            # Top
            if y + rad > 1:
                forces[i, 1] -= (y + rad - 1) * 0.5
        
        centers += alpha * forces
        
        # Project back to [0, 1] just in case, though forces handle it
        # But we need to respect radius buffer.
        # Actually, the LP will fix radii later. Here we just keep centers in [0,1].
        centers[:, 0] = np.clip(centers[:, 0], 0, 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, 1)

    return centers

def generate_hex_grid(n):
    """Generate a hexagonal grid of points."""
    # Estimate radius for n circles in hex packing
    # Area approx n * pi * r^2 <= 1 -> r <= 1/sqrt(n*pi)
    # But packing density ~ 0.9
    # r approx 0.1
    r_est = 0.105
    
    # Spacing
    dx = 2 * r_est
    dy = r_est * math.sqrt(3)
    
    points = []
    row = 0
    while len(points) < n:
        y = r_est + row * dy
        if y + r_est > 1:
            break
            
        # Offset alternate rows
        x_start = r_est + (0.5 if row % 2 == 1 else 0) * dx # Shift by r_est? No, dx/2 = r_est
        
        # Wait, standard hex:
        # Row 0: x = r, 3r, 5r... (dx = 2r)
        # Row 1: x = 2r, 4r, 6r... (shifted by r)
        # My dx is 2r_est. Shift is r_est.
        
        if row % 2 == 1:
            x_start = r_est + r_est # Shift by r_est? No.
            # If row 0 starts at r_est.
            # Row 1 should start at 2*r_est?
            # Distance between (r, 0) and (2r, h) is sqrt(r^2 + h^2).
            # h = r*sqrt(3). dist = sqrt(r^2 + 3r^2) = 2r. Correct.
            x_start = 2 * r_est
        else:
            x_start = r_est
            
        x = x_start
        while x + r_est <= 1:
            points.append([x, y])
            x += dx
            if len(points) >= n:
                break
        row += 1
        
    return np.array(points[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Strategy: Multiple restarts with local optimization
    best_centers = None
    best_radii = None
    max_sum_r = 0.0
    
    # Try multiple random seeds and initial configurations
    for seed in range(50):
        np.random.seed(seed)
        
        # Initial centers
        # Mix of hex grid and random
        if seed < 20:
            centers = generate_hex_grid(n)
            # Add small noise
            centers += np.random.normal(0, 0.005, centers.shape)
            centers = np.clip(centers, 0, 1)
        else:
            # Random packing initialization
            centers = np.random.uniform(0.1, 0.9, (n, 2))
            
        # Initial radii guess (small)
        radii = np.full(n, 0.05)
        
        # Iterative improvement loop
        for step in range(10):
            # 1. Relax centers to remove overlaps based on current radii
            centers = relax_centers(centers, radii, iterations=50)
            
            # 2. Solve LP for max radii given centers
            radii = solve_radii_lp(centers)
            
            # Check validity
            if np.any(radii < -1e-9):
                continue
            
            current_sum = np.sum(radii)
            
            # 3. If we have room, try to perturb to find better layout
            # Only perturb if we are not at a perfect fit or just randomly
            if step < 8: # Don't perturb in last step
                # Perturb centers slightly
                centers += np.random.normal(0, 0.002, centers.shape)
                centers = np.clip(centers, 0, 1)
                
        # Final check
        # Validate
        try:
            # Manual quick check
            valid = True
            # Boundary
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x < -r or x > 1+r or y < -r or y > 1+r:
                    valid = False
                    break
            # Overlap
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
            
            if valid and current_sum > max_sum_r:
                max_sum_r = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        except:
            pass

    # Fallback if nothing worked (should not happen)
    if best_centers is None:
        best_centers = generate_hex_grid(n)
        best_radii = solve_radii_lp(best_centers)
        max_sum_r = np.sum(best_radii)

    return best_centers, best_radii, max_sum_r