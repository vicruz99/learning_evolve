# sol_000313 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a46c309d) state=74c06096 sum of radii=1.502105 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Define the LP solver for fixed centers
    def solve_radii(centers):
        n = centers.shape[0]
        c_obj = np.ones(n) * -1  # Maximize sum -> Minimize negative sum
        
        A_ub = []
        b_ub = []
        
        # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
        # r_i <= x_i  =>  r_i <= x_i
        # r_i <= 1-x_i => r_i <= 1-x_i
        # ...
        for i in range(n):
            x, y = centers[i]
            # r_i <= x
            row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(x)
            # r_i <= 1-x
            row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(1-x)
            # r_i <= y
            row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(y)
            # r_i <= 1-y
            row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(1-y)
            
        # 2. Pairwise constraints: r_i + r_j <= dist(i,j)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None) for _ in range(n)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                # Radii are negated back
                return np.array(res.x) 
            else:
                # Fallback to small radii if infeasible (should not happen with r>=0)
                return np.full(n, 0.001)
        except Exception:
            return np.full(n, 0.001)

    def get_score(centers):
        radii = solve_radii(centers)
        return np.sum(radii), radii

    def generate_hexagonal_init():
        # Try to fit 26 points in a hexagonal grid
        # Base hex grid
        points = []
        # Heuristic to find spacing
        # Try different spacings
        for scale in np.linspace(0.05, 0.2, 10):
            pts = []
            # Generate lattice points
            # Rows
            for j in range(6):
                y = j * scale * np.sqrt(3)
                for i in range(6):
                    x = i * scale * 2
                    if j % 2 == 1: x += scale
                    pts.append([x, y])
            
            pts = np.array(pts)
            # Filter inside [0,1]
            mask = (pts[:,0] >= 0) & (pts[:,0] <= 1) & (pts[:,1] >= 0) & (pts[:,1] <= 1)
            pts_in = pts[mask]
            
            if len(pts_in) >= n:
                # Select first n
                points.append(pts_in[:n])
        
        if points:
            return points[0]
        
        # Fallback: Random
        return np.random.rand(n, 2)

    # Perform multiple random restarts with hill climbing
    num_restarts = 5
    for _ in range(num_restarts):
        # Initialize centers
        centers = generate_hexagonal_init()
        # Add small random noise
        centers += np.random.randn(*centers.shape) * 0.01
        centers = np.clip(centers, 0, 1)

        current_sum, current_radii = get_score(centers)
        
        # Hill Climbing
        step_size = 0.1
        for iter_step in range(200):
            # Generate neighbor
            perturbation = np.random.randn(*centers.shape) * step_size
            new_centers = centers + perturbation
            new_centers = np.clip(new_centers, 0, 1)
            
            new_sum, new_radii = get_score(new_centers)
            
            if new_sum > current_sum:
                current_sum = new_sum
                current_radii = new_radii
                centers = new_centers
            else:
                # If no improvement, reduce step size occasionally
                pass
            
            # Decay step size
            step_size *= 0.995
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers
            best_radii = current_radii

    # Final validation check
    radii = solve_radii(best_centers)
    # Re-evaluate sum
    final_sum = np.sum(radii)
    
    return best_centers, radii, final_sum

# Helper function to ensure we don't use closures
def solve_radii_standalone(centers):
    n = centers.shape[0]
    c_obj = np.ones(n) * -1
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(x)
        row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(1-x)
        row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(y)
        row = np.zeros(n); row[i] = 1; A_ub.append(row); b_ub.append(1-y)
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.array(res.x) 
        else:
            return np.full(n, 0.001)
    except Exception:
        return np.full(n, 0.001)

# Re-define run_packing to use standalone function to avoid closure issues if strict
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    def get_score(centers):
        radii = solve_radii_standalone(centers)
        return np.sum(radii), radii

    def generate_hexagonal_init():
        points = []
        for scale in np.linspace(0.05, 0.2, 10):
            pts = []
            for j in range(6):
                y = j * scale * np.sqrt(3)
                for i in range(6):
                    x = i * scale * 2
                    if j % 2 == 1: x += scale
                    pts.append([x, y])
            pts = np.array(pts)
            mask = (pts[:,0] >= 0) & (pts[:,0] <= 1) & (pts[:,1] >= 0) & (pts[:,1] <= 1)
            pts_in = pts[mask]
            if len(pts_in) >= n:
                points.append(pts_in[:n])
        if points:
            return points[0]
        return np.random.rand(n, 2)

    num_restarts = 5
    for _ in range(num_restarts):
        centers = generate_hexagonal_init()
        centers += np.random.randn(*centers.shape) * 0.01
        centers = np.clip(centers, 0, 1)

        current_sum, current_radii = get_score(centers)
        
        step_size = 0.1
        for iter_step in range(200):
            perturbation = np.random.randn(*centers.shape) * step_size
            new_centers = centers + perturbation
            new_centers = np.clip(new_centers, 0, 1)
            
            new_sum, new_radii = get_score(new_centers)
            
            if new_sum > current_sum:
                current_sum = new_sum
                current_radii = new_radii
                centers = new_centers
            
            step_size *= 0.995
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers
            best_radii = current_radii

    radii = solve_radii_standalone(best_centers)
    final_sum = np.sum(radii)
    
    return best_centers, radii, final_sum
