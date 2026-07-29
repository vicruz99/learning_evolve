# sol_000260 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fd8f28d8) state=1d931b64 sum of radii=2.541421 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    
    # --- 1. Initialization: Hexagonal-like Grid ---
    # We generate a set of points in a hexagonal pattern and pick the first 26
    # that fit within the square with a small initial radius.
    
    def generate_hex_centers(n, r_init=0.05):
        centers = []
        # Hexagonal lattice parameters
        # Row height
        h = math.sqrt(3) * r_init
        # Col width
        w = 2 * r_init
        
        y = r_init
        row_idx = 0
        while len(centers) < n:
            x = r_init + (row_idx % 2) * w/2
            while x < 1.0 - r_init and len(centers) < n:
                if 0 <= x <= 1 and 0 <= y <= 1:
                    centers.append([x, y])
                x += w
            y += h
            row_idx += 1
        return np.array(centers[:n])

    # Start with a grid to ensure good coverage, then perturb
    # A 5x5 grid has 25 points. We need 26.
    # Let's start with a perturbed grid + hexagonal influence
    centers = np.zeros((n_circles, 2))
    idx = 0
    
    # 5x5 Grid
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.2, 0.1 + j * 0.2]
            idx += 1
    
    # 26th circle in a gap
    centers[25] = [0.2, 0.2] # Center of a 2x2 block
    
    # Initial small radii to start
    radii = np.ones(n_circles) * 0.01

    # --- 2. Optimization Loop ---
    
    def solve_lp(centers):
        """
        Given fixed centers, find max radii using LP.
        Maximize sum(r_i)
        s.t. r_i + r_j <= dist(i, j)
             r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
             r_i >= 0
        """
        n = len(centers)
        c_obj = -np.ones(n) # Minimize -sum(r) => Maximize sum(r)
        
        A_ub = []
        b_ub = []
        
        # Distance constraints: r_i + r_j <= d_ij
        # Only check pairs that are relatively close to save computation
        # Actually for 26, 325 pairs is fine.
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
        
        # Boundary constraints: r_i <= margin
        for i in range(n):
            x, y = centers[i]
            margin = min(x, 1 - x, y, 1 - y)
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(margin)
            
        # Non-negativity is handled by bounds in linprog
        bounds = [(0, None) for _ in range(n)]
        
        # Solve
        # method='highs' is efficient
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return -res.fun, res.x
        else:
            return 0.0, np.zeros(n)

    def get_pressure(centers, radii):
        """
        Calculate forces to move centers apart if they are tight.
        If r_i + r_j == dist(i,j), they are touching. 
        We want to increase dist to allow larger radii.
        """
        forces = np.zeros_like(centers)
        n = len(centers)
        
        # Check pairs
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                sum_r = radii[i] + radii[j]
                
                # If they are touching or very close
                if dist < sum_r + 1e-4: # Allow tiny tolerance
                    # Force proportional to how much they violate or are tight
                    # We want to push them apart.
                    # Direction: i <- j
                    if dist > 1e-9:
                        vec = (centers[i] - centers[j]) / dist
                        # Magnitude of force. If tight, push.
                        # If we are strictly satisfying constraints, dist >= sum_r.
                        # But to optimize, we look at "slack". 
                        # Slack = dist - sum_r. We want to increase slack.
                        # Actually, this LP based approach finds max radii for fixed centers.
                        # To improve, we need to move centers to increase the LP value.
                        # A simple heuristic: if a constraint is active (tight), move centers apart.
                        
                        # Let's apply a repulsive force if they are "bound"
                        # Since LP maximizes radii, the active constraints are the bottlenecks.
                        # If r_i + r_j = dist, moving i and j apart increases capacity.
                        
                        force_mag = 0.01 # Small step
                        forces[i] += vec * force_mag
                        forces[j] -= vec * force_mag

        # Boundary pressure
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # If hitting wall
            if x - r < 1e-4: forces[i, 0] += 0.01 # Push right
            if x + r > 1 - 1e-4: forces[i, 0] -= 0.01 # Push left
            if y - r < 1e-4: forces[i, 1] += 0.01 # Push up
            if y + r > 1 - 1e-4: forces[i, 1] -= 0.01 # Push down
            
        return forces

    best_sum = 0.0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Main Iterative Refinement
    # We will run a local search.
    
    for iteration in range(100): # Number of macro steps
        
        # 1. Solve LP to get best radii for current centers
        current_sum, current_radii = solve_lp(centers)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = current_radii.copy()
            
        # 2. Calculate forces and move centers
        forces = get_pressure(centers, current_radii)
        centers += forces
        
        # Project back to [0,1]x[0,1] just in case
        centers = np.clip(centers, 0.01, 0.99)
        
        # 3. Random Jitter to escape local minima
        # As iterations progress, reduce jitter
        jitter = 0.02 * (1.0 - iteration/100.0)
        if jitter > 0.001:
            noise = np.random.randn(n_circles, 2) * jitter
            centers += noise
            centers = np.clip(centers, 0.01, 0.99)

    # Final LP solve on the best centers found
    final_sum, final_radii = solve_lp(best_centers)
    
    # One last check with a few random restarts on the best centers
    # to see if we can nudge it further
    for _ in range(20):
        temp_centers = best_centers + np.random.randn(n_circles, 2) * 0.005
        temp_centers = np.clip(temp_centers, 0.01, 0.99)
        s, r = solve_lp(temp_centers)
        if s > final_sum:
            final_sum = s
            final_radii = r
            best_centers = temp_centers

    return best_centers, final_radii, final_sum
