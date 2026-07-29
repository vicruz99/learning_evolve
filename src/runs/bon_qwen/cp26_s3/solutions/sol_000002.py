# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3ad176de) state=b14f8cee sum of radii=2.606407 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses numerical optimization (SLSQP) starting from a hexagonal-like grid.
    """
    n_circles = 26
    
    # --- Initialization ---
    # We try to initialize in a pattern close to optimal hexagonal packing.
    # A rough estimate for radius is 0.1.
    # Let's try to fit 26 circles in a 5x5 grid (25) + 1, or a hexagonal layout.
    # Hexagonal layout: rows shifted.
    
    # Let's create a grid of points and select 26, or generate specific layout.
    # A simple grid perturbation is robust.
    # Let's try a 6x5 grid (30 points) and pick 26, or just 5x5 + 1.
    # Actually, a random valid initialization might get stuck, so grid is better.
    
    # Try to place centers in a 5x5 grid structure but slightly randomized or
    # better, a hexagonal pattern.
    # Let's generate a hexagonal lattice subset.
    # Spacing dx = 0.22, dy = 0.22 * sqrt(3)/2 approx 0.19
    # But we don't know exact r yet. Let's use relative positions.
    
    # Let's use a simple grid initialization first for robustness, 
    # then rely on optimizer to move them.
    # 26 circles. 5x5 = 25. 
    # Let's place 5 rows of 5 circles? 25 circles.
    # Where is the 26th? Maybe squeeze in.
    # Let's try a 4x7 grid? 28 circles. Remove 2?
    # Let's just scatter them in a grid that fits comfortably (small r).
    
    # Initialization: 26 points in a 6x5 grid (30 slots), take first 26.
    # Grid size 1x1.
    # 6 columns: x = 1/12, 3/12, ... 11/12 ?
    # 5 rows: y = 1/10, 3/10, ... 9/10 ?
    
    # Let's do a denser grid initialization to help the optimizer find dense packing.
    # But they must not overlap initially.
    # Let's set initial radius to 0.05 (diameter 0.1).
    # Grid spacing 0.2 should work.
    
    # Let's create a 5x6 grid (30 points) with spacing 0.2
    # x in [0.1, 0.9], y in [0.1, 0.9]
    # 5 columns: 0.1, 0.3, 0.5, 0.7, 0.9
    # 6 rows: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6? No, need 0.9 limit.
    # 0.1, 0.25, 0.4, 0.55, 0.7, 0.85?
    
    # Let's just use a random valid initialization within [0.2, 0.8]
    # and small radii.
    
    np.random.seed(42) # For reproducibility
    
    # Better initialization: Hexagonal packing pattern
    # Approximate optimal radius 0.101.
    # Let's place centers on a lattice that fits 26 circles.
    # Pattern:
    # Row 0: 5 circles
    # Row 1: 6 circles
    # Row 2: 5 circles
    # Row 3: 6 circles
    # Row 4: 4 circles
    # Total 26.
    # This looks like a hexagonal subset.
    
    centers_init = []
    r_init = 0.05 # Start small to avoid overlap
    
    # Coordinates for hexagonal pattern
    # We need to fit in [0,1]x[0,1].
    # Let's normalize coordinates to [0.1, 0.9] roughly.
    
    # Row 0: 5 circles
    # x coords: 0.2, 0.4, 0.6, 0.8, 1.0? No, must be <= 0.9 + r
    # Let's space them evenly in [0.15, 0.85]
    
    def add_row(y, count, shift_x=0):
        # Spread 'count' circles in x range [0.15, 0.85]
        # With shift for hex pattern
        if count == 0: return
        # Calculate x positions
        # Total width available ~ 0.7
        # Spacing 0.7 / (count - 1) if count > 1
        # Or just linspace
        xs = np.linspace(0.15, 0.85, count)
        if shift_x > 0:
            xs = xs + shift_x
            # Wrap or clip? Just clip to [0.15, 0.85] range roughly
            # Actually better to just shift center
            pass 
        for x in xs:
            # Keep within bounds [0.1, 0.9]
            cx = np.clip(x, 0.1, 0.9)
            centers_init.append([cx, y])

    # Try to construct a dense packing layout manually
    # 5 rows
    # Row y=0.15: 5 circles
    # Row y=0.35: 6 circles (shifted)
    # Row y=0.55: 5 circles
    # Row y=0.75: 6 circles (shifted)
    # Row y=0.95: 4 circles (centered) -> wait y=0.95 is too high for r=0.05
    # Max y should be 0.9.
    
    # Let's just use a grid of 26 points.
    # 5 columns, 6 rows? 30 points.
    # Take 26.
    x_coords = np.linspace(0.1, 0.9, 5)
    y_coords = np.linspace(0.1, 0.9, 6)
    # 5x6 = 30 points.
    grid_points = []
    for y in y_coords:
        for x in x_coords:
            grid_points.append([x, y])
    
    # Take first 26 points
    centers_init = np.array(grid_points[:26])
    radii_init = np.full(26, 0.05)
    
    # --- Optimization ---
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    # Total variables: 3 * 26 = 78
    # Flatten centers and radii
    initial_params = np.concatenate([centers_init.flatten(), radii_init])
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5] (actually r <= 0.5)
    # But tighter bounds: r <= 0.5, x,y in [0,1]
    # Actually x >= r, x <= 1-r etc are handled by constraints, 
    # but box bounds help.
    # Let's set box bounds for x, y in [0, 1] and r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n_circles) + [(0.0, 0.5)] * n_circles
    
    def objective(params):
        # Maximize sum of radii -> Minimize negative sum
        radii = params[2 * n_circles:]
        return -np.sum(radii)
    
    def get_centers_radii(params):
        centers = params[:2 * n_circles].reshape((n_circles, 2))
        radii = params[2 * n_circles:]
        return centers, radii

    # Constraints
    constraints = []
    
    # 1. Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # x_i - r_i >= 0
    # 1 - (x_i + r_i) >= 0
    # y_i - r_i >= 0
    # 1 - (y_i + r_i) >= 0
    
    for i in range(n_circles):
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[2*i] - p[2*n_circles + i]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[2*i] - p[2*n_circles + i]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[2*i + 1] - p[2*n_circles + i]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[2*i + 1] - p[2*n_circles + i]
        })

    # 2. Non-overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # dist^2 - (r_i + r_j)^2 >= 0
    
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: 
                    (p[2*i] - p[2*j])**2 + (p[2*i + 1] - p[2*j + 1])**2 - (p[2*n_circles + i] + p[2*n_circles + j])**2
            })
            
    # Run optimization
    # SLSQP is good for constrained non-linear problems
    try:
        res = opt.minimize(
            objective, 
            initial_params, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        final_params = res.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial if failed (though unlikely with valid init)
        final_params = initial_params

    centers = final_params[:2 * n_circles].reshape((n_circles, 2))
    radii = final_params[2 * n_circles:]
    
    sum_radii = np.sum(radii)
    
    # Just in case, enforce non-negative radii and valid bounds strictly
    # (Optimizer should handle, but numerical errors might occur)
    radii = np.maximum(radii, 0.0)
    
    # Clip centers to valid range relative to radii?
    # The validate function checks strict bounds.
    # We can slightly shrink radii if they touch boundary exactly to be safe?
    # But constraints are >= 0, so x-r >= 0 is satisfied.
    # However, floating point might result in x-r = -1e-16.
    # Let's adjust slightly if needed.
    
    # Check validity
    # If invalid, we might need to shrink.
    # But with 1e-12 tolerance in validation, it should be fine.
    
    return centers, radii, sum_radii

# Helper to run and check (for local testing, though not required in final output structure)
if __name__ == "__main__":
    # We can't run this inside the provided block easily without imports
    # But the function is ready.
    pass
