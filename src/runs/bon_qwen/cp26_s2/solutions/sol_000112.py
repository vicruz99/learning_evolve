# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 028484b6) state=17fb1a4d sum of radii=2.505421 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Helper to generate a hexagonal grid initialization
    def get_hex_init(seed=0):
        np.random.seed(seed)
        # Try to fit 26 circles in a hexagonal pattern
        # Approximate spacing
        # Area ~ 1, n=26, density ~ 0.9 -> r ~ 0.105
        # Let's create a grid of points and pick 26
        
        # We want to cover the square.
        # Let's generate a dense hexagonal grid and select points
        # Or just place them manually in rows
        
        # Configuration: Rows with varying counts to fit 26
        # Pattern: 5, 6, 5, 6, 4 (Total 26)
        # Or 5, 5, 6, 5, 5 (Total 26)
        # Let's try 5 rows
        
        rows_counts = [5, 6, 5, 6, 4]
        centers = []
        
        # Initial guess for radius to scale positions
        # If r=0.1, width for 6 circles is 1.2 (too big). 
        # We will scale later, but let's just place them in [0,1]
        # using a heuristic spacing.
        
        y_spacing = 1.0 / (len(rows_counts) + 1) # rough
        # Better: distribute y evenly
        y_coords = np.linspace(0.15, 0.85, len(rows_counts))
        
        for i, count in enumerate(rows_counts):
            # x spacing depends on count
            # Distribute count circles in [0, 1]
            # For hexagonal, odd rows shifted
            x_coords = np.linspace(0.15 + (0.15/count), 0.85 - (0.15/count), count) # naive
            # Actually, let's just use linspace for initial placement
            # It doesn't have to be perfect, optimizer will fix it.
            x_coords = np.linspace(1.0/(count+1), 1.0 * count/(count+1), count)
            
            # Shift for hexagonal effect
            shift = 0.5 / count # approx half spacing
            if i % 2 == 1:
                x_coords = x_coords + shift
                # Clip to bounds
                x_coords = np.clip(x_coords, 0, 1)
            
            for x in x_coords:
                centers.append([x, y_coords[i]])
        
        centers = np.array(centers)
        
        # Add some random noise to avoid symmetry traps
        noise = np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers + noise, 0.05, 0.95)
        
        return centers

    # Vectorized constraint functions
    # Variables vector V: [x1...x26, y1...y26, r1...r26]
    # But let's use equal radii first for robustness, then maybe relax?
    # The prompt allows variable radii. Let's try variable radii.
    # Variables: x (26), y (26), r (26). Total 78.
    
    # To save time and complexity, let's stick to EQUAL radii first.
    # If equal radii sum is high, it's a valid solution.
    # Maximizing sum of radii with variable radii is harder to converge.
    # However, "sum of radii" optimization with variable radii often converges to equal radii 
    # for regular packings.
    # Let's optimize for a single radius variable 'r' shared by all.
    # Variables: x (26), y (26), r (1). Total 53.
    
    def objective(V):
        # V[0:26] -> x
        # V[26:52] -> y
        # V[52] -> r
        # We want to maximize sum(r_i) = 26 * r
        # Minimize -26 * r
        return -26.0 * V[52]

    def constraints_eq(V):
        x = V[:26]
        y = V[26:52]
        r = V[52]
        
        # Boundary constraints
        # x >= r  => x - r >= 0
        # x <= 1-r => 1 - x - r >= 0
        # same for y
        
        # Vectorized boundary constraints
        # We return a flat array of constraint values
        c = []
        c.append(x - r)       # 26 constraints
        c.append(1.0 - x - r) # 26 constraints
        c.append(y - r)       # 26 constraints
        c.append(1.0 - y - r) # 26 constraints
        
        # Pairwise overlap constraints
        # dist(i,j)^2 >= (2r)^2
        # (xi-xj)^2 + (yi-yj)^2 - 4r^2 >= 0
        
        # Compute pairwise squared distances
        # Using broadcasting
        # X diff: (26, 1) - (1, 26) -> (26, 26)
        X = x[:, None]
        Y = y[:, None]
        
        D2 = (X - X.T)**2 + (Y - Y.T)**2
        
        # We only need upper triangle (i < j)
        # Create mask
        mask = np.triu(np.ones((26, 26), dtype=bool), k=1)
        
        # Extract upper triangle values
        overlap_vals = D2[mask] - (2.0 * r)**2
        
        c.append(overlap_vals)
        
        return np.concatenate(c)

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(26):
        bounds.append((0.0, 1.0)) # x
    for _ in range(26):
        bounds.append((0.0, 1.0)) # y
    bounds.append((0.0, 0.5))     # r

    best_V = None
    best_obj = float('inf')

    # Try multiple initializations
    for seed in range(10):
        try:
            centers_init = get_hex_init(seed=seed)
            # Initial radius guess
            # Estimate based on grid
            min_dist = np.min(cdist(centers_init, centers_init)) # dist to self is 0
            # compute min dist between distinct points
            dists = []
            for i in range(26):
                for j in range(i+1, 26):
                    dists.append(np.linalg.norm(centers_init[i] - centers_init[j]))
            r_init = 0.5 * min(dists) if dists else 0.05
            
            # Construct initial vector
            V0 = np.concatenate([centers_init[:, 0], centers_init[:, 1], [r_init]])
            
            # Constraints for SLSQP
            cons = ({'type': 'ineq', 'fun': constraints_eq})
            
            res = minimize(objective, V0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < best_obj:
                # Validate the solution manually to be sure
                x_sol = res.x[:26]
                y_sol = res.x[26:52]
                r_sol = res.x[52]
                
                centers_sol = np.column_stack((x_sol, y_sol))
                radii_sol = np.full(26, r_sol)
                
                # Quick validation check
                valid = True
                # Check bounds
                if np.any(x_sol < r_sol - 1e-9) or np.any(x_sol > 1.0 - r_sol + 1e-9):
                    valid = False
                if np.any(y_sol < r_sol - 1e-9) or np.any(y_sol > 1.0 - r_sol + 1e-9):
                    valid = False
                
                # Check overlaps
                if valid:
                    dists_sq = cdist(centers_sol, centers_sol, metric='sqeuclidean')
                    # diagonal is 0
                    np.fill_diagonal(dists_sq, np.inf)
                    min_dist_sq = np.min(dists_sq)
                    if min_dist_sq < (2.0 * r_sol)**2 - 1e-9:
                        valid = False
                
                if valid:
                    current_sum = 26.0 * r_sol
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = centers_sol
                        best_radii = radii_sol
                        best_obj = res.fun

        except Exception as e:
            print(f"Optimization failed for seed {seed}: {e}")
            continue

    # Fallback or refinement if best is not good
    # If we didn't find a good packing, maybe try variable radii?
    # But for 26 circles, equal is likely best.
    # Let's try one more refinement with variable radii if equal sum is low?
    # Actually, let's just ensure we return a valid result.
    
    if best_centers is None:
        # Default fallback: small circles in grid
        best_centers = np.random.uniform(0.2, 0.8, (26, 2))
        best_radii = np.full(26, 0.01)
        best_sum_radii = 0.26

    return best_centers, best_radii, best_sum_radii

# Function to validate locally (just for debugging, not used in return)
def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0: return False
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

# Execution logic if run directly, though run_packing is the entry point
if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    print(f"Valid: {validate_packing(centers, radii)}")
