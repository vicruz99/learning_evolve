# sol_000215 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 722eaafb) state=90eb8d57 sum of radii=1.909454 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def calculate_optimal_radii(centers):
    """
    Given fixed centers, solves the LP to maximize sum of radii.
    Variables: r_1, ..., r_26
    Objective: Max sum(r_i)  ->  Min -sum(r_i)
    Constraints:
      1. r_i >= 0
      2. r_i <= x_i
      3. r_i <= 1 - x_i
      4. r_i <= y_i
      5. r_i <= 1 - y_i
      6. r_i + r_j <= dist(i, j)  => r_i + r_j - dist(i, j) <= 0
    """
    n = len(centers)
    
    # Objective function: minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints matrix A_ub * r <= b_ub
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x_i  => r_i <= x_i
        row = [0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1 - x_i => r_i <= 1 - x_i
        row = [0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - x)
        
        # r_i <= y_i
        row = [0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1 - y_i
        row = [0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - y)
        
    # Non-overlap constraints: r_i + r_j <= dist_ij
    # Precompute distances
    dists = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d > 1e-9: # Avoid degenerate cases if centers are identical
                row = [0] * n
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
                
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for r_i: 0 <= r_i
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using high precision method if available, otherwise default
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback to small radii if LP fails (should not happen with valid centers)
        return np.full(n, 0.0)

def generate_hexagonal_centers(n_circles):
    """
    Generates initial centers in a hexagonal grid pattern.
    """
    centers = []
    # Estimate grid size
    # Area per circle approx 2*sqrt(3)*r^2. 
    # Let's just try to pack them roughly in a square.
    
    # Try to find a grid dimension
    # Rows and cols
    # Approx n = rows * cols
    # Let's try to fill a rectangle
    
    r_guess = 0.1 # Initial guess
    row_step = r_guess * math.sqrt(3)
    col_step = 2 * r_guess
    
    # Heuristic: try different numbers of rows
    best_centers = None
    best_score = -1
    
    for rows in range(1, 10):
        cols = int(np.ceil(n_circles / rows))
        # Generate centers
        current_centers = []
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= n_circles:
                    break
                x = c * col_step
                y = r * row_step
                if r % 2 == 1:
                    x += r_guess # Shift odd rows
                current_centers.append([x, y])
                count += 1
            if count >= n_circles:
                break
        
        if len(current_centers) == n_circles:
            # Normalize to fit in [0,1]x[0,1]
            cx = np.array(current_centers)
            if np.max(cx[:,0]) > 0:
                cx[:,0] /= np.max(cx[:,0])
            if np.max(cx[:,1]) > 0:
                cx[:,1] /= np.max(cx[:,1])
            
            # Check validity (rough)
            # Calculate min distance
            dists = np.linalg.norm(cx[:, np.newaxis] - cx, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            # Objective: we want centers to be spread out? 
            # Actually, for LP, having centers further apart helps.
            # But we normalized to [0,1], so this is fixed.
            # Let's just pick the one that fits best?
            # Actually, normalization distorts the shape.
            # Better to scale the grid to fit in square tightly.
            
            # Let's just return one reasonable configuration
            # A roughly square aspect ratio is good.
            aspect = np.max(cx[:,0]) / np.max(cx[:,1])
            if 0.8 < aspect < 1.2:
                return cx

    # Fallback to random if heuristic fails
    return np.random.rand(n_circles, 2)

def run_packing():
    n = 26
    np.random.seed(42)
    
    # 1. Initial centers
    centers = generate_hexagonal_centers(n)
    
    # 2. Optimization loop (Simulated Annealing)
    # We will perturb centers and re-solve LP for radii
    
    # Initial radii
    radii = calculate_optimal_radii(centers)
    current_sum = np.sum(radii)
    best_sum = current_sum
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    temp = 0.1 # Initial temperature
    alpha = 0.99 # Cooling rate
    iterations = 2000
    
    for i in range(iterations):
        # Generate new centers by perturbing current centers
        new_centers = centers.copy()
        # Random perturbation
        perturbation = np.random.uniform(-0.05, 0.05, size=new_centers.shape)
        new_centers += perturbation
        
        # Clip to [0,1] to keep valid
        new_centers = np.clip(new_centers, 0, 1)
        
        # Solve for optimal radii with new centers
        new_radii = calculate_optimal_radii(new_centers)
        new_sum = np.sum(new_radii)
        
        # Acceptance probability
        delta = new_sum - current_sum
        if delta > 0:
            accept = True
        else:
            if temp > 1e-6:
                prob = math.exp(delta / temp)
                accept = np.random.random() < prob
            else:
                accept = False
        
        if accept:
            centers = new_centers
            radii = new_radii
            current_sum = new_sum
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        
        # Cool down
        temp *= alpha
        
        # Small chance to reset or large jump? 
        # Just standard cooling
        
    # 3. Final validation and cleanup
    # Ensure no NaNs or invalid states
    # Recalculate radii one last time to be sure
    final_radii = calculate_optimal_radii(best_centers)
    
    # Clip radii to be non-negative just in case
    final_radii = np.maximum(final_radii, 0)
    
    # Validate constraints manually to be safe
    # Check boundaries
    for i in range(n):
        x, y = best_centers[i]
        r = final_radii[i]
        # Hard constraints from LP should hold, but let's clamp if needed
        # Actually LP ensures r <= x etc.
        # But floating point might be slightly off?
        # The validation function allows 1e-12 tolerance.
        
    # Check overlaps
    # If any overlap, we might need to reduce radii slightly.
    # But LP ensures r_i + r_j <= dist.
    
    return best_centers, final_radii, np.sum(final_radii)

# Helper to make sure we don't use closures
def validate_and_return(centers, radii):
    # Just a wrapper to match the signature if needed, 
    # but run_packing is the main entry.
    return centers, radii, np.sum(radii)
