# sol_000152 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e234a3e4) state=d17fc735 sum of radii=2.615411 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def evaluate_constraints(x):
    """
    Computes the constraints for the circle packing problem.
    Returns an array of values that must be <= 0 for a valid packing.
    
    Constraints:
    1. r_i - x_i <= 0
    2. r_i - y_i <= 0
    3. x_i + r_i - 1 <= 0
    4. y_i + r_i - 1 <= 0
    5. (r_i + r_j)^2 - ((x_i - x_j)^2 + (y_i - y_j)^2) <= 0
    """
    n = len(x) // 3
    centers = x[:2*n].reshape((n, 2))
    radii = x[2*n:]
    
    cons = []
    
    # Boundary constraints
    # Vectorized operations for efficiency
    cons.extend(radii - centers[:, 0])      # r - x <= 0
    cons.extend(radii - centers[:, 1])      # r - y <= 0
    cons.extend(centers[:, 0] + radii - 1.0)# x + r - 1 <= 0
    cons.extend(centers[:, 1] + radii - 1.0)# y + r - 1 <= 0
    
    # Overlap constraints
    # Check all pairs (i, j) with i < j
    for i in range(n):
        ri = radii[i]
        xi, yi = centers[i]
        for j in range(i + 1, n):
            rj = radii[j]
            xj, yj = centers[j]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            
            r_sum = ri + rj
            r_sum_sq = r_sum * r_sum
            
            # Constraint: (r_i + r_j)^2 <= dist^2  =>  (r_i + r_j)^2 - dist^2 <= 0
            cons.append(r_sum_sq - dist_sq)
            
    return np.array(cons)

def constraints_ineq(x):
    """
    Wrapper for evaluate_constraints to satisfy scipy.optimize.minimize 
    inequality constraint requirement (fun(x) >= 0).
    Since evaluate_constraints returns values <= 0, we negate them.
    """
    return -evaluate_constraints(x)

def objective_func(x):
    """
    Objective function to minimize.
    We want to maximize sum of radii, so we minimize negative sum.
    """
    n = len(x) // 3
    radii = x[2*n:]
    return -np.sum(radii)

def run_packing():
    """
    Runs the optimization to pack 26 circles in a unit square.
    """
    n = 26
    dim = 3 * n
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # List of initial configurations
    starts = []
    rng = np.random.default_rng(42)
    
    # 1. Random starts
    for _ in range(5):
        centers = rng.uniform(0.1, 0.9, size=(n, 2))
        radii = np.full(n, 0.05)
        starts.append(np.concatenate([centers.flatten(), radii]))
        
    # 2. Grid start (5x6 subset)
    pts = []
    for r in range(5):
        y = (r + 0.5) / 5.0
        for c in range(6):
            x = (c + 0.5) / 6.0
            pts.append([x, y])
    pts = np.array(pts[:n])
    radii = np.full(n, 0.05)
    starts.append(np.concatenate([pts.flatten(), radii]))
    
    # 3. Hexagonal start
    pts_hex = []
    for r in range(5):
        y = (r + 0.5) / 5.0
        # Shift every other row
        shift = 0.5 / 6.0 if r % 2 == 1 else 0.0
        for c in range(6):
            x = (c + 0.5 + shift) / 6.0
            if 0 <= x <= 1:
                pts_hex.append([x, y])
    pts_hex = np.array(pts_hex[:n])
    radii = np.full(n, 0.05)
    starts.append(np.concatenate([pts_hex.flatten(), radii]))
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Run optimization for each start
    for i, x0 in enumerate(starts):
        try:
            res = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints_ineq},
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or (-res.fun > best_sum):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2*n].reshape((n, 2))
                    best_radii = res.x[2*n:]
        except Exception as e:
            print(f"Optimization failed on start {i}: {e}")
            continue
            
    # If best_sum is still low, it might be stuck. 
    # But with multiple starts, it should be okay.
    
    return best_centers, best_radii, best_sum

# Validation code from prompt (read-only)
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    import numpy as np # Ensure numpy is available
    n = centers.shape[0]

    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

# To test locally
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Validation: {validate_packing(centers, radii)}")
