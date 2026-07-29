# sol_000202 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7f4d5c4f) state=dff79913 sum of radii=2.531603 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def solve_radii(centers):
    """
    Given fixed centers, solve for optimal radii to maximize sum of radii.
    Constraints:
    1. r_i >= 0
    2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i (inside square)
    3. r_i + r_j <= distance(i, j) for all i < j
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    # Bounds for r_i: 0 <= r_i <= min_dist_to_boundary
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        bounds.append((0, max_r))
        
    # Constraints: r_i + r_j <= dist_ij
    # A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.fun * -1, res.x
        else:
            return 0, np.zeros(n)
    except Exception:
        return 0, np.zeros(n)

def get_hex_grid_points(n_points, square_size=1.0):
    """
    Generate points on a hexagonal grid inside the square.
    We try to fit as many as possible, then select n_points.
    """
    points = []
    # Estimate radius/r to determine grid spacing
    # For 26 points, roughly 5x5. 
    # Hex spacing roughly 2*r. 
    # Let's just generate a dense grid and pick best?
    # Or generate specific rows.
    
    # Let's try rows with 5 and 6 items
    # 6, 5, 6, 5, 4 -> 26
    # Or 5, 5, 5, 5, 6 -> 26
    
    # Let's generate a standard hex grid covering the square
    # Spacing dx, dy
    # To fit ~26, area per point ~ 1/26 ~ 0.038.
    # Hex area ~ sqrt(3)/2 * side^2. side ~ 2r.
    # Let's just create a grid with density high enough.
    
    # Simple approach: grid search
    # But we need a good start for optimization.
    
    # Let's place them in rows
    rows = []
    # Try to fit 26 circles
    # Row 0: 6 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 6 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 4 circles
    # Total 26.
    
    # Let's determine vertical spacing. 
    # 5 rows. Height 1. 
    # y coords: r, r+h, r+2h, r+3h, r+4h.
    # Top constraint: r+4h+r <= 1 => 2r + 4h <= 1.
    # h approx sqrt(3) * r.
    # 2r + 4*1.732*r = 8.9r <= 1 => r <= 0.11.
    # But row of 6 requires width.
    # 6 circles in hex row (shifted relative to neighbors, but aligned in row?)
    # In a single row, centers separated by 2r.
    # Width for 6 circles: 11r (center to center 10r + radii).
    # 11r + r <= 1 => 12r <= 1 => r <= 0.0833.
    # So if we have a row of 6, r is small.
    
    # Maybe better to have rows of 5?
    # 5, 5, 5, 5, 6? No, 6 is wide.
    # 5, 5, 5, 5, 5, 1?
    # 5 rows of 5 = 25. Add 1.
    # 5x5 grid allows r=0.1.
    # We can perturb 5x5 grid.
    
    centers = np.zeros((26, 2))
    
    # Base 5x5 grid
    r_base = 0.095 # slightly less than 0.1 to allow movement
    count = 0
    for i in range(5):
        for j in range(5):
            if count < 26:
                x = (i + 0.5) * 0.2 # centered at 0.1, 0.3... no, 0.1+0.1=0.2?
                # Grid points: 0.1, 0.3, 0.5, 0.7, 0.9
                x = 0.1 + i * 0.2
                y = 0.1 + j * 0.2
                centers[count] = [x, y]
                count += 1
    
    # The 26th circle
    # Place it in a gap.
    # Gap at (0.2, 0.2) in the grid of (0.1, 0.1)...
    # Wait, centers are 0.1, 0.3...
    # Midpoint is 0.2.
    # Distance to neighbors (0.1, 0.1) is sqrt(0.1^2 + 0.1^2) = 0.1414.
    # Sum of radii r + r_base <= 0.1414.
    # If r_base=0.095, r <= 0.046.
    # Let's place it there.
    if count < 26:
        centers[count] = [0.2, 0.2]
        count += 1
        
    # Add some random noise to break symmetry and help optimization
    noise = np.random.uniform(-0.005, 0.005, centers.shape)
    centers = centers + noise
    # Clip to valid range [0.01, 0.99] to avoid boundary issues initially
    centers = np.clip(centers, 0.02, 0.98)
    
    return centers

def optimize_packing(centers, iterations=500):
    """
    Optimize centers to maximize sum of radii.
    Uses a simple hill climbing with random perturbations.
    """
    best_centers = centers.copy()
    current_sum, _ = solve_radii(best_centers)
    best_sum = current_sum
    
    step_size = 0.01
    
    for k in range(iterations):
        # Perturb a random circle
        idx = np.random.randint(0, 26)
        
        # Try moving in random direction
        move = np.random.uniform(-step_size, step_size, 2)
        new_centers = best_centers.copy()
        new_centers[idx] += move
        
        # Keep inside square (loosely)
        new_centers[idx] = np.clip(new_centers[idx], 0.001, 0.999)
        
        s, _ = solve_radii(new_centers)
        
        if s > best_sum:
            best_centers = new_centers
            best_sum = s
            # Occasionally reduce step size
            if k % 100 == 0:
                step_size *= 0.9
        else:
            # If not improving, maybe reduce step size or try different strategy
            # But simple hill climbing might get stuck.
            pass
            
    # Final refinement with smaller steps
    step_size = 0.001
    for k in range(500):
        idx = np.random.randint(0, 26)
        move = np.random.uniform(-step_size, step_size, 2)
        new_centers = best_centers.copy()
        new_centers[idx] += move
        new_centers[idx] = np.clip(new_centers[idx], 0.001, 0.999)
        
        s, _ = solve_radii(new_centers)
        if s > best_sum:
            best_centers = new_centers
            best_sum = s
            
    final_sum, radii = solve_radii(best_centers)
    return best_centers, radii, final_sum

def run_packing():
    # Set seed for reproducibility if needed, but random is fine for exploration
    np.random.seed(42) 
    
    # Initial centers
    centers = get_hex_grid_points(26)
    
    # Optimize
    centers, radii, sum_radii = optimize_packing(centers, iterations=1000)
    
    return centers, radii, sum_radii

# To verify locally
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(r)}, Max radius: {np.max(r)}")
    
    # Basic validation
    n = c.shape[0]
    valid = True
    for i in range(n):
        x, y = c[i]
        rad = r[i]
        if x < rad or x > 1-rad or y < rad or y > 1-rad:
            valid = False
            print(f"Circle {i} out of bounds")
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((c[i]-c[j])**2))
            if dist < r[i] + r[j] - 1e-9:
                valid = False
                print(f"Overlap {i} {j}")
    print(f"Valid: {valid}")
