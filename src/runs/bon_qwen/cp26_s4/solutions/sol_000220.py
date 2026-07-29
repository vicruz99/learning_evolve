# sol_000220 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 76d635d8) state=ffa565a3 sum of radii=2.487174 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generates an optimized packing of 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    def penalty_and_grad(x, radii):
        """
        Computes the sum of squared penetration depths (penalty) and its gradient.
        x: flattened array of (x, y) coordinates for n_circles.
        radii: array of radii (assumed equal for simplicity in this optimization).
        """
        centers = x.reshape(-1, 2)
        penalty = 0.0
        grad = np.zeros_like(x)
        r = radii[0]
        
        # Check boundary constraints
        for i in range(n_circles):
            cx, cy = centers[i]
            # Left
            if cx < r:
                diff = r - cx
                penalty += diff**2
                grad[2*i] += -2 * diff
            # Right
            if cx + r > 1.0:
                diff = cx + r - 1.0
                penalty += diff**2
                grad[2*i] += 2 * diff
            # Bottom
            if cy < r:
                diff = r - cy
                penalty += diff**2
                grad[2*i + 1] += -2 * diff
            # Top
            if cy + r > 1.0:
                diff = cy + r - 1.0
                penalty += diff**2
                grad[2*i + 1] += 2 * diff
                
        # Check pairwise constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = 2.0 * r
                
                # Avoid division by zero
                if dist_sq < 1e-12:
                    dist = 1e-6
                    dx_norm = 0.0
                    dy_norm = 0.0
                else:
                    dist = math.sqrt(dist_sq)
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                
                if dist < min_dist:
                    diff = min_dist - dist
                    penalty += diff**2
                    
                    # Gradient: d/da ( (2r - |a-b|)^2 )
                    # chain rule: 2 * diff * (- d/da |a-b| )
                    # d/da |a-b| is unit vector from b to a
                    factor = -2.0 * diff
                    
                    grad[2*i] += factor * dx_norm
                    grad[2*i + 1] += factor * dy_norm
                    grad[2*j] += -factor * dx_norm
                    grad[2*j + 1] += -factor * dy_norm
                    
        return penalty, grad

    def generate_hexagonal_grid(seed=0):
        """Generates a hexagonal grid layout for centers, scaled to fit in [0,1]"""
        rng = np.random.default_rng(seed)
        
        # Estimate radius for 26 circles in hex packing
        # Area approx 1. Density ~0.9. 26 * pi * r^2 * 0.9 = 1 -> r ~ 0.11
        # Start with a smaller radius to ensure fit
        r_init = 0.08 
        
        # Create a grid of potential points
        points = []
        # Hex spacing
        dx = 2 * r_init
        dy = math.sqrt(3) * r_init
        
        y = r_init
        while y + r_init < 1.0:
            x = r_init
            # Stagger rows
            row_offset = (len(points) // max(1, int(1/dx) + 1)) % 2 * r_init # Approx stagger
            # Better stagger logic:
            # If row index (based on y) is odd, shift x by r_init
            row_idx = int((y - r_init) / dy)
            shift = r_init if row_idx % 2 == 1 else 0
            
            x = r_init + shift
            while x + r_init < 1.0:
                points.append([x, y])
                x += dx
            y += dy
            
        # If we have more points than needed, select the first n_circles
        # If fewer, we need to adjust, but hex grid usually generates enough
        if len(points) < n_circles:
             # Fallback to random if grid is too small (unlikely with r=0.08)
             return np.random.uniform(r_init, 1-r_init, size=(n_circles, 2))
            
        # Shuffle and take n_circles
        rng.shuffle(points)
        selected = np.array(points[:n_circles])
        
        # Add small random noise
        noise = rng.uniform(-0.01, 0.01, size=selected.shape)
        selected = np.clip(selected + noise, r_init, 1-r_init)
        
        return selected.flatten()

    # Optimization Loop
    # We iterate on radius, optimizing positions for each step
    # Or better: Optimize positions for a fixed radius, then increase radius.
    
    current_r = 0.05
    max_r = 0.15 # Upper bound estimate
    
    # Initial positions
    current_positions = generate_hexagonal_grid(42)
    
    # We will try to fit circles of radius r. 
    # We perform a binary search or step-up approach on r.
    # For each r, we minimize the penalty function.
    
    step = 0.001
    while current_r < max_r:
        # Optimize positions for current radius
        for _ in range(3): # Multiple restarts or refinements
            res = minimize(
                fun=lambda x: penalty_and_grad(x, [current_r])[0],
                x0=current_positions,
                method='L-BFGS-B',
                jac=lambda x: penalty_and_grad(x, [current_r])[1],
                bounds=[(current_r, 1-current_r)] * (2 * n_circles),
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            current_positions = res.x
            if res.fun < 1e-5:
                break
        
        # If penalty is low enough, try increasing radius
        if penalty_and_grad(current_positions, [current_r])[0] < 1e-4:
            current_r += step
        else:
            # If stuck, try to perturb and retry, or break if we can't increase
            # Simple back-off or just continue with smaller step
            current_r -= step/2
            step = step / 2
            
        if step < 1e-6:
            break

    final_r = current_r - step # Use the last successful radius
    
    # Final refinement with the determined radius
    res = minimize(
        fun=lambda x: penalty_and_grad(x, [final_r])[0],
        x0=current_positions,
        method='L-BFGS-B',
        jac=lambda x: penalty_and_grad(x, [final_r])[1],
        bounds=[(final_r, 1-final_r)] * (2 * n_circles),
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    final_centers = res.x.reshape(-1, 2)
    final_radii = np.full(n_circles, final_r)
    
    # Check if we can squeeze a bit more by optimizing radii individually?
    # For this task, equal radii is a very strong baseline.
    # However, to strictly maximize sum, we can try to expand the smallest gaps.
    # But given the constraints and time, the equal radius hex packing is robust.
    
    # Just in case, let's verify validity and adjust if needed
    if not validate_packing(final_centers, final_radii):
        # Fallback: shrink radii slightly until valid
        while not validate_packing(final_centers, final_radii) and final_radii[0] > 0.01:
            final_radii -= 0.0001
            
    return final_centers, final_radii, np.sum(final_radii)

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False

    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative
    for i in range(n):
        if radii[i] < 0:
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True
