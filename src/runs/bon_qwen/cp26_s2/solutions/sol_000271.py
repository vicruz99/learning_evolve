# sol_000271 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9068c8d6) state=ef671fc3 sum of radii=2.454842 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def generate_hexagonal_points(n_points):
    """Generates a grid of points in a hexagonal pattern."""
    # Estimate number of points per row/col
    # Approx density is pi / sqrt(12). 
    # Side length L approx sqrt(n * sqrt(12) / pi).
    # For unit square, n points -> diameter ~ 1/sqrt(n).
    # Let's just generate a grid large enough and select best.
    
    # Try to generate a grid covering roughly 0 to 1
    # We will create a grid with spacing roughly 1/sqrt(n)
    # and then pick the best n points.
    
    # Heuristic: create a triangular grid.
    # Rows and Cols estimation
    rows = int(math.sqrt(n_points * 1.5)) + 2
    cols = rows
    
    points = []
    y_step = math.sqrt(3)/2 * 0.2 # Initial guess for spacing
    x_step = 0.2
    
    # We will optimize these later, just generating candidates
    # Let's make a dense grid
    # Actually, better to just return a structured grid centered in square
    
    # Let's generate a triangular lattice
    # x = i * d + (j % 2) * 0.5 * d
    # y = j * d * math.sqrt(3)/2
    
    d = 0.15 # Diameter guess
    points = []
    for j in range(int(1/d * 1.2) + 5):
        for i in range(int(1/d * 1.2) + 5):
            x = i * d + (j % 2) * 0.5 * d
            y = j * d * math.sqrt(3)/2
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # Shuffle to pick random subset? No, we want to pick 26 that fit best.
    # But for initialization, just taking first 26 is okay if we optimize.
    # Better: Sort by distance to center? Or just take first n.
    return np.array(points[:max(n_points, len(points))])

def get_constraints(centers, radii):
    """
    Returns a list of constraint violation values.
    Positive value indicates violation.
    We want to minimize sum of violations to 0.
    """
    n = len(radii)
    violations = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # Violation: r - x > 0 etc.
        violations.append(max(0, r - x))
        violations.append(max(0, r - (1 - x)))
        violations.append(max(0, r - y))
        violations.append(max(0, r - (1 - y)))
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            sum_r = radii[i] + radii[j]
            violations.append(max(0, sum_r - dist))
            
    return violations

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal packing of equal circles
    # Start with a reasonable radius estimate
    # Area of 26 circles ~ 1. r^2 ~ 1/(26*pi) ~ 0.012. r ~ 0.11.
    # But packing efficiency ~ 0.9. r ~ 0.105.
    
    # Generate initial centers
    # Let's use a perturbed hexagonal grid
    r_init = 0.09
    centers = np.zeros((n, 2))
    
    # Place in a roughly 5x5 or 6x5 grid
    # 5 rows of 5 circles = 25. Need 1 more.
    # Let's try 5 rows: 5, 5, 5, 5, 6?
    # Or 6 rows: 5, 4, 5, 4, 5, 3?
    
    # Let's just place them in a hexagonal pattern and scale/fit
    # Hexagonal coordinates
    # x = i * 2r + (j%2)*r
    # y = j * r * sqrt(3)
    
    # We will use an optimizer to find the best equal radius packing first
    # Variables: r (scalar), dx, dy (shifts), rotation?
    # Actually, just optimizing centers directly is easier.
    
    # Initial random valid centers
    centers = np.random.uniform(r_init, 1-r_init, size=(n, 2))
    radii = np.full(n, r_init)
    
    # Simple repulsion to resolve overlaps initially
    for _ in range(100):
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 2 * r_init:
                    vec = centers[i] - centers[j]
                    vec /= (dist + 1e-9)
                    centers[i] += vec * (r_init - dist/2)
                    centers[j] -= vec * (r_init - dist/2)
        
        # Boundary push
        for i in range(n):
            x, y = centers[i]
            r = r_init
            if x < r: centers[i, 0] = r
            if x > 1-r: centers[i, 0] = 1-r
            if y < r: centers[i, 1] = r
            if y > 1-r: centers[i, 1] = 1-r

    # 2. Optimization using Scipy
    # We maximize sum(radii)
    # Variables: centers (26*2) and radii (26) -> 78 variables.
    # But radii are coupled.
    # Strategy: Optimize centers to maximize the minimum feasible radius for equal circles.
    # Then allow radii to vary.
    
    # Phase 1: Maximize equal radius r
    # Variables: centers (52 variables). r is derived.
    # r_i = min(dist to boundary, min_j dist_ij / 2)
    # We want to maximize min(r_i).
    
    def objective_equal_radius(x):
        centers = x.reshape(-1, 2)
        # Calculate max possible equal radius
        # r <= dist to boundary
        r_boundary = np.minimum(
            np.minimum(centers[:, 0], 1 - centers[:, 0]),
            np.minimum(centers[:, 1], 1 - centers[:, 1])
        )
        # r <= dist to neighbor / 2
        # This is expensive to compute inside loop, but n=26 is small.
        # We can just return negative of min(r_boundary) as a proxy?
        # No, we must consider neighbors.
        
        # Compute pairwise distances
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        
        r_neighbor = min_dists / 2
        
        # The feasible radius for each circle given centers
        r_feasible = np.minimum(r_boundary, r_neighbor)
        
        # We want to maximize the minimum feasible radius (for equal circles)
        # Or sum of feasible radii?
        # Let's maximize sum of feasible radii.
        return -np.sum(r_feasible)

    # Initial guess for centers from previous repulsion step
    x0 = centers.flatten()
    
    bounds = [(0.05, 0.95) for _ in range(2*n)] # Keep centers away from edges initially
    
    # Use L-BFGS-B
    res = opt.minimize(objective_equal_radius, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
    
    best_centers = res.x.reshape(-1, 2)
    
    # Compute radii based on best_centers
    # Now we allow radii to be different to squeeze more sum
    # Current radii are constrained by geometry
    r_boundary = np.minimum(
        np.minimum(best_centers[:, 0], 1 - best_centers[:, 0]),
        np.minimum(best_centers[:, 1], 1 - best_centers[:, 1])
    )
    dists = np.linalg.norm(best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    # r_i + r_j <= dist_ij
    # This is a linear programming problem now?
    # Max sum r_i s.t. r_i + r_j <= dist_ij, r_i <= boundary_i
    
    # We can solve this LP
    # Variables r_0 ... r_25
    # Maximize sum r
    # Constraints: r_i + r_j <= dist_ij for all i<j
    # r_i <= r_boundary_i
    # r_i >= 0
    
    from scipy.optimize import linprog
    
    c = -np.ones(n) # Minimize -sum r
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
        # Boundary constraints
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(r_boundary[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds_r = [(0, None) for _ in range(n)]
    
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    
    if res_lp.success:
        radii_opt = res_lp.x
    else:
        # Fallback to equal radii based on geometry
        r_feasible = np.minimum(r_boundary, np.min(dists, axis=1)/2)
        radii_opt = r_feasible

    # Final validation and slight adjustment if needed
    # Check validity
    centers_final = best_centers
    radii_final = radii_opt
    
    # Ensure non-negative
    radii_final = np.maximum(radii_final, 0)
    
    # Clip centers to valid range just in case
    # (Though LP should have respected boundaries, centers are fixed here)
    # Wait, if radii are large, centers might be invalid?
    # The LP constraint r_i <= boundary_i ensures that if we place circle at center with radius r_i, it fits.
    # So centers_final is valid with radii_final.
    
    sum_r = np.sum(radii_final)
    
    return centers_final, radii_final, sum_r

# Helper to run and validate locally (not part of submission)
if __name__ == "__main__":
    centers, radii, total_r = run_packing()
    print(f"Sum of radii: {total_r}")
    
    # Check validity manually
    valid = True
    n = 26
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            print(f"Boundary violation: {i}")
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                print(f"Overlap: {i}, {j}")
    
    print(f"Valid: {valid}")
