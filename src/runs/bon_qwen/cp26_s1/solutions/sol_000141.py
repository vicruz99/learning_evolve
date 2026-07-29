# sol_000141 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fa534539) state=7d654858 sum of radii=2.262428 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def solve_radii(centers):
    """
    Solves the LP to find optimal radii for fixed centers.
    Maximizes sum(r) subject to boundary and non-overlap constraints.
    """
    n = centers.shape[0]
    if n == 0:
        return 0.0, np.array([])

    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)

    # Constraints A_ub @ r <= b_ub
    A_ub = []
    b_ub = []

    # 1. Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(x)

        # r_i <= 1 - x
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - x)

        # r_i <= y
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(y)

        # r_i <= 1 - y
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - y)

    # 2. Pairwise constraints: r_i + r_j <= dist(c_i, c_j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]

    try:
        # Use 'highs' method for efficiency
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback if LP fails, return small radii
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def calculate_forces(centers, radii):
    """
    Calculates heuristic forces on centers to improve packing.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Strength constants
    repulsion_strength = 0.05
    boundary_strength = 0.05
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            
            if dist < 1e-9:
                continue
                
            r_sum = radii[i] + radii[j]
            
            # If circles are touching or overlapping, push apart
            # We look at slack. If slack is small, force is high.
            # However, in LP solution, they might be exactly touching.
            # We want to increase distance to allow larger radii.
            # The force should be proportional to how much the constraint limits the sum.
            # Heuristic: if dist approx r_sum, push.
            
            # Normalize direction
            direction = diff / dist
            
            # Force magnitude could depend on radii. Larger circles might need more space?
            # Simple repulsion
            forces[i] += direction * repulsion_strength
            forces[j] -= direction * repulsion_strength

    # Boundary forces
    # If a circle is constrained by a wall (r approx distance to wall), 
    # moving center away from wall increases allowable radius.
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Check left wall (x - r approx 0)
        # If r is close to x, x is limiting. Increase x (push right).
        if r > x - 1e-4:
            forces[i, 0] += boundary_strength
            
        # Check right wall (1 - x - r approx 0)
        # If r is close to 1-x, 1-x is limiting. Decrease x (push left).
        if r > (1.0 - x) - 1e-4:
            forces[i, 0] -= boundary_strength
            
        # Check bottom wall (y - r approx 0)
        if r > y - 1e-4:
            forces[i, 1] += boundary_strength
            
        # Check top wall (1 - y - r approx 0)
        if r > (1.0 - y) - 1e-4:
            forces[i, 1] -= boundary_strength
            
    return forces

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Grid pattern
    # Try to distribute points evenly. 
    # sqrt(26) approx 5.1. 5x5 grid is 25 points.
    # We can place 25 in a grid and 1 in a gap.
    
    centers = np.zeros((n, 2))
    
    # Create a 5x5 grid
    # Spacing 0.2, offset 0.1
    # Points: 0.1, 0.3, 0.5, 0.7, 0.9
    pts = []
    for r in range(5):
        for c in range(5):
            pts.append([0.1 + c * 0.2, 0.1 + r * 0.2])
    
    # We have 25 points. Add 26th.
    # A good spot might be between points. 
    # E.g., (0.2, 0.2) is a midpoint of grid cells? 
    # Grid centers are at 0.1, 0.3. Midpoint 0.2.
    pts.append([0.2, 0.2]) 
    
    centers = np.array(pts[:n])
    
    # Add small random noise to break symmetry
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to valid range initially
    centers = np.clip(centers, 0.01, 0.99)

    best_sum = -1.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)

    # 2. Optimization Loop
    # We perform a local search using forces derived from the LP solution.
    iterations = 300
    
    for step in range(iterations):
        # Solve LP for current centers
        current_sum, radii = solve_radii(centers)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
        
        # Calculate forces to move centers
        forces = calculate_forces(centers, radii)
        
        # Learning rate / step size, decaying over time
        alpha = 0.1 * (1.0 / (1.0 + step * 0.02))
        
        # Add random noise to escape local minima
        noise_scale = 0.05 * (1.0 / (1.0 + step * 0.05))
        noise = np.random.normal(0, noise_scale, centers.shape)
        
        # Update centers
        centers += alpha * forces + noise
        
        # Enforce boundaries strictly to keep centers valid
        # Centers must be in [0, 1]. 
        # LP handles radius constraints, but centers must be valid coords.
        centers = np.clip(centers, 0.001, 0.999)

    # Final solve to ensure consistency and get final radii
    final_sum, final_radii = solve_radii(best_centers)
    
    # Ensure radii are valid for the returned centers
    # (LP guarantees this, but just in case)
    # The LP solved for best_centers gives radii that satisfy constraints.
    
    return best_centers, final_radii, final_sum

# To allow standalone execution or import
if __name__ == "__main__":
    # Just to test locally if needed, though the platform calls run_packing
    pass
