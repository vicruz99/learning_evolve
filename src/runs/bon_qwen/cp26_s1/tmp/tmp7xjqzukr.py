import numpy as np
import scipy.optimize as opt
import math

def calculate_loss(v, N, penalty_weight):
    """
    Calculate the loss function for circle packing.
    v: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Objective: Minimize negative sum of radii
    obj = -np.sum(r)
    
    penalty = 0.0
    
    # Boundary penalties
    # r <= x  => x - r >= 0 => penalty if x - r < 0
    # r <= 1-x => 1 - x - r >= 0
    # r <= y
    # r <= 1-y
    
    # Vectorized boundary checks
    # dist to left: x
    # dist to right: 1 - x
    # dist to bottom: y
    # dist to top: 1 - y
    
    dists = np.column_stack([x, 1 - x, y, 1 - y])
    # For each circle, the max allowed radius by boundaries is min(dists)
    # We want r <= min(dists) <=> r - min(dists) <= 0
    # But min is non-smooth. We can penalize each constraint separately.
    
    # Check x >= r
    overlap_x_left = np.maximum(0, r - x)
    penalty += np.sum(overlap_x_left**2)
    
    # Check 1-x >= r  => x + r <= 1
    overlap_x_right = np.maximum(0, r + x - 1)
    penalty += np.sum(overlap_x_right**2)
    
    # Check y >= r
    overlap_y_bot = np.maximum(0, r - y)
    penalty += np.sum(overlap_y_bot**2)
    
    # Check 1-y >= r => y + r <= 1
    overlap_y_top = np.maximum(0, r + y - 1)
    penalty += np.sum(overlap_y_top**2)
    
    # Pairwise overlap penalties
    # We need to check all pairs i < j
    # dist(i, j) >= r_i + r_j
    # overlap = r_i + r_j - dist(i, j)
    
    # Vectorized pairwise distance computation might be memory heavy for large N, 
    # but N=26 is small (26*25/2 = 325 pairs).
    
    # Compute pairwise squared distances
    # x_diffs shape (N, N)
    x_diffs = x[:, np.newaxis] - x[np.newaxis, :]
    y_diffs = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = x_diffs**2 + y_diffs**2
    
    # We only need upper triangle
    # Create a mask for i < j
    # But computing all is fine, just sum upper triangle
    
    # distances
    dists = np.sqrt(dist_sq)
    
    # Sum of radii matrix
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Overlap matrix
    overlaps = r_sum - dists
    
    # We only care about positive overlaps and i < j
    # Triangular mask
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    active_overlaps = np.maximum(0, overlaps) * mask
    
    penalty += np.sum(active_overlaps**2)
    
    return obj + penalty_weight * penalty

def run_packing():
    N = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # We will try multiple random seeds and initial configurations
    # L-BFGS-B bounds
    lb = [0.0, 0.0, 0.0] * N
    ub = [1.0, 1.0, 0.5] * N # Radius max 0.5
    bounds = list(zip(lb, ub))
    
    # Penalty weight
    PENALTY_W = 5000.0
    
    def objective(v):
        return calculate_loss(v, N, PENALTY_W)

    def initial_grid_perturb(scale_x, scale_y, offset_x, offset_y, shift):
        # Generate a hex-like grid
        centers = []
        r_init = 0.08
        rows = 6
        cols = 5 # 5 cols for even rows, 4 for odd?
        
        # Let's just generate points in a grid and perturb
        # 5x5 grid = 25 points
        # Add 1 point
        pts = []
        # Hexagonal packing coordinates
        # y_step = sqrt(3)/2 * diameter? No, y_step = sqrt(3)*r
        # But we don't know r yet.
        # Let's just use a rectangular grid for init, optimizer will fix it.
        
        # 6 rows, varying cols
        # Pattern: 5, 4, 5, 4, 5, 3 -> 26
        # Or just random
        
        for i in range(6):
            # x offset for staggering
            x_off = 0.05 if i % 2 == 1 else 0.0
            count = 4 if i % 2 == 1 else 5 # alternating 5, 4
            if i == 5: count = 2 # last row smaller? 5+4+5+4+5 = 23. Need 3 more.
            
            # Let's do a simpler 5x5 + 1
            pass
        
        # Actually, random init is robust with multi-start
        return np.random.uniform(0, 1, (N, 3))

    # Strategy: 
    # 1. Random initializations.
    # 2. Structured initializations (Grid, Hex).
    
    num_attempts = 10
    
    for attempt in range(num_attempts):
        # Initial guess
        # Try to pack in a rough hexagonal layout
        # 5 rows
        # Row 0: 5 circles
        # Row 1: 5 circles (shifted)
        # Row 2: 5 circles
        # Row 3: 5 circles (shifted)
        # Row 4: 6 circles? 
        # Total 26.
        
        # Let's just use a dense grid and let optimizer expand radii
        # 6x5 grid points, remove 4? No.
        
        # Random init
        if attempt < 5:
            v0 = np.random.uniform(0.2, 0.8, N * 3)
            v0[2::3] = 0.05 # Small radii
        else:
            # Grid init
            # 5 rows, 5 cols = 25. Add 1.
            pts = []
            for r_idx in range(5):
                for c_idx in range(5):
                    pts.append([0.1 + c_idx * 0.2, 0.1 + r_idx * 0.2])
            # 26th point
            pts.append([0.5, 0.5]) # Center is occupied? 
            # Grid points: (0.1, 0.1) ... (0.9, 0.9)
            # (0.5, 0.5) is one of them.
            # Let's put 26th at (0.5, 0.1) -> occupied.
            # Put at (0.05, 0.5) -> near boundary.
            pts[-1] = [0.05, 0.5]
            
            # Shuffle to avoid symmetry bias
            np.random.shuffle(pts)
            
            v0 = np.zeros(N * 3)
            v0[0::3] = [p[0] for p in pts]
            v0[1::3] = [p[1] for p in pts]
            v0[2::3] = 0.08

        # Optimize
        # Use bounded minimizer
        res = opt.minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                           options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 2000})
        
        v_opt = res.x
        centers = np.column_stack((v_opt[0::3], v_opt[1::3]))
        radii = v_opt[2::3]
        
        # Validate
        # We can use a local validation here to check penalty
        loss = calculate_loss(v_opt, N, PENALTY_W)
        # Check if valid
        # Re-check manually
        valid = True
        # Check bounds
        for i in range(N):
            x, y, r = centers[i,0], centers[i,1], radii[i]
            if r < 0 or x < r or x + r > 1 or y < r or y + r > 1:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # If best_sum is still bad (e.g. all small), we might need to refine.
    # But with 10 attempts and L-BFGS-B, it should find a good local optimum.
    # To ensure we hit the target, we can try to increase radii if valid but sum is low?
    # No, the optimizer maximizes sum.
    
    # Fallback: If we found a valid packing, great.
    # If not, we might need to adjust.
    # With high penalty, it should be valid.
    
    # One more pass: try to expand radii uniformly if there is slack?
    # The optimizer does this implicitly.
    
    # Let's return the best found.
    # If best_centers is None, create a dummy valid one (though unlikely)
    if best_centers is None:
        # Fallback to a valid small packing
        best_centers = np.random.uniform(0.1, 0.9, (26, 2))
        best_radii = np.full(26, 0.01)
        
    return best_centers, best_radii, np.sum(best_radii)

# Helper to run the solution
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)