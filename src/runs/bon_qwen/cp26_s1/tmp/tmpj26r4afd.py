import numpy as np
import scipy.optimize as opt
import math

def get_hex_grid_centers(n, margin=0.0):
    """
    Generate initial centers based on a hexagonal lattice packing.
    n: number of circles
    """
    # Estimate radius to determine spacing. 
    # For 26 circles, roughly 5x5 or 6x5 grid. 
    # Hexagonal packing is denser.
    # Let's try to fit them in.
    
    # Approximate number of rows
    rows = int(math.ceil(n / 6)) # heuristic
    
    # Better approach: just place them in a dense pattern
    # Start with a 6x5 grid (30 spots) and pick 26 best?
    # Or just generate hex grid.
    
    # Let's generate a hex grid that fills the square
    # Spacing s. 
    # Width of row with k circles: 2r + (k-1)*sqrt(3)*r ? No, distance is 2r?
    # Hex grid: dx = sqrt(3)*r, dy = 1.5*r? 
    # Actually standard hex packing:
    # centers at (x, y). 
    # Row 0: y = r, x = r, r+2r, ...
    # Row 1: y = r + sqrt(3)r, x = r+r, r+r+2r ...
    
    # We don't know r yet. Let's assume r ~ 0.1.
    r_est = 0.1
    dy = math.sqrt(3) * r_est
    dx = 2 * r_est
    
    centers = []
    y = r_est
    while len(centers) < n:
        x = r_est
        # Check if we need to shift x for odd rows?
        # In standard hex packing, rows are offset by r.
        # But here we are placing points, not circles of fixed size yet.
        # Let's just use a skewed grid.
        
        row_circles = []
        while x <= 1 - r_est and len(centers) < n:
            row_circles.append([x, y])
            x += dx
            # If we want hex packing, every other row should be shifted?
            # Actually, for point placement, a shifted grid is fine.
        
        if len(row_circles) > 0:
            # If this is an odd row (1-indexed), maybe shift?
            # Let's apply a shift for odd indices to make it hex-like
            # But simpler: just dense packing.
            # Let's try to fit as many as possible.
            centers.extend(row_circles)
        
        y += dy
        if len(centers) < n:
            # Shift next row?
            # If we just add dx, it's a square grid (stretched).
            # Hex grid requires offset.
            # Let's alternate offset?
            # Actually, simple dense grid is often enough to start optimization.
            # But let's try a shifted row.
            # Shift by dx/2 ?
            # centers_in_row = math.floor((1 - 2*r_est)/dx) + 1
            # If we shift, we might lose one circle on edge.
            pass

    # If we didn't get enough, fill with random or grid
    if len(centers) < n:
        # Fallback to random or grid
        pass
    
    return np.array(centers[:n])

def compute_max_radii(centers):
    """
    Given centers, compute the maximum radii such that circles don't overlap and stay in bounds.
    This is solved by iteratively clamping radii.
    We want to maximize sum of radii.
    Constraints:
    r_i <= x_i
    r_i <= 1-x_i
    r_i <= y_i
    r_i <= 1-y_i
    r_i + r_j <= dist(i, j)
    
    This is equivalent to finding the largest r vector in a polytope.
    We can use a simple iterative relaxation or linear programming.
    For speed and simplicity in this context, we can use a 'push' method or just scipy linprog.
    Linprog is robust.
    """
    n = centers.shape[0]
    c = centers
    
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r_i) => Minimize -sum(r_i)
    # f = -1 vector
    
    f = -np.ones(n)
    
    # Constraints: A_ub @ r <= b_ub
    # 1. Boundary constraints: r_i <= c[i, 0], r_i <= 1-c[i, 0], etc.
    # 2. Pairwise constraints: r_i + r_j <= dist_ij
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(c[i, 0])
        
        # r_i <= 1-x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - c[i, 0])
        
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(c[i, 1])
        
        # r_i <= 1-y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - c[i, 1])
        
    # Pairwise constraints
    # r_i + r_j <= dist_ij
    # To keep matrix small, we only add constraints for reasonably close pairs?
    # Actually for 26 circles, 325 pairs is fine.
    dists = np.sqrt(((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Use high_p=False for speed? Or just standard.
    # Method 'highs' is default in newer scipy, 'simplex' or 'interior-point'.
    try:
        res = opt.linprog(f, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # If LP fails (infeasible?), return small radii
            # Should not happen if centers are valid (distinct)
            return np.full(n, 1e-5)
    except Exception:
        return np.full(n, 1e-5)

def optimize_packing():
    """
    Main optimization routine.
    We will use a randomized local search / hill climbing.
    We optimize centers to maximize sum of radii (computed by LP).
    """
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        # 1. Generate initial centers
        # Try hexagonal-ish random perturbation
        # Base grid
        n = 26
        
        # Create a 6x5 grid (30 points) and remove 4? 
        # Or 5x6?
        # Let's just place points on a grid and jitter.
        # 5 rows, ~5-6 cols.
        
        # Hexagonal packing layout
        # Rows
        ys = np.linspace(0.1, 0.9, 5) # 5 rows
        xs = np.linspace(0.1, 0.9, 6) # 6 cols -> 30 points
        
        # We need 26 points.
        # Let's take 6, 5, 6, 5, 4 ? Or just random selection?
        # A dense hexagonal packing usually alternates row lengths.
        # Row lengths: 6, 5, 6, 5, 4 sums to 26.
        # Let's try to construct that.
        
        candidates = []
        # Row 0: 6 circles
        for i in range(6):
            candidates.append([i * 0.175 + 0.05, 0.1]) # x spacing approx 1/6? No.
            # Better: center them.
            # 6 circles in width 1: spacing 1/5 = 0.2? No, diameter.
            # Centers at 0.1, 0.3, 0.5, 0.7, 0.9 for 5 circles.
            # For 6 circles, centers at 0.083, 0.25, 0.41, 0.58, 0.75, 0.91?
            # Let's just use a regular grid and let optimizer fix it.
        
        # Let's use a dense random initialization constrained to bounds
        # Or a grid.
        # Grid 5x6 (30 points) -> pick 26.
        # Or 6x5.
        
        # Let's try a hexagonal lattice explicitly.
        # Spacing s = 0.12 (approx)
        centers = []
        row = 0
        while len(centers) < 26:
            y = 0.1 + row * 0.18 # vertical spacing
            # Check if y is in bounds
            if y > 0.9: 
                # try smaller spacing or break?
                # If we can't fit, just place what we can?
                # But we need 26.
                # Let's just use a simple grid for initialization.
                pass
            
            # Horizontal offset for hex packing
            offset = 0.09 if row % 2 == 1 else 0.0
            x = 0.1 + offset
            while x < 0.9 and len(centers) < 26:
                centers.append([x, y])
                x += 0.18
            row += 1
        
        if len(centers) < 26:
            # Fallback to random
            centers = np.random.rand(26, 2)
        
        # Jitter
        centers = centers + np.random.randn(26, 2) * 0.01
        # Clip to [0,1]
        centers = np.clip(centers, 0.05, 0.95) # Keep away from boundary initially? 
        # Actually boundary allows touching.
        
        # Optimization Loop (Simple Gradient Ascent / Hill Climbing)
        # We move centers to increase the sum of radii.
        # Since we can compute radii via LP, we can estimate gradient numerically or just use perturbations.
        
        current_centers = centers.copy()
        current_radii = compute_max_radii(current_centers)
        current_sum = np.sum(current_radii)
        
        # Local search
        # Randomly perturb one center at a time?
        # Or all?
        # Given the LP is expensive, maybe simple coordinate ascent.
        
        for step in range(50): # 50 iterations
            improved = False
            # Try to perturb each center
            for i in range(26):
                best_local_sum = current_sum
                best_local_c = current_centers[i].copy()
                
                # Try random moves
                for _ in range(10):
                    delta = np.random.randn(2) * 0.02 * (1 + step/10) # Decrease step size?
                    new_c = current_centers[i] + delta
                    new_c = np.clip(new_c, 0, 1)
                    
                    # Check if valid (not too close to others? LP handles it, but LP might fail if points overlap exactly)
                    # To avoid LP failure, ensure points are distinct.
                    # If points coincide, distance is 0, radii must be 0.
                    
                    temp_centers = current_centers.copy()
                    temp_centers[i] = new_c
                    
                    # Check for exact duplicates (dist < 1e-9)
                    is_valid = True
                    for j in range(26):
                        if i != j:
                            d = np.linalg.norm(temp_centers[i] - temp_centers[j])
                            if d < 1e-6:
                                is_valid = False
                                break
                    
                    if is_valid:
                        radii = compute_max_radii(temp_centers)
                        s = np.sum(radii)
                        if s > best_local_sum:
                            best_local_sum = s
                            best_local_c = new_c.copy()
                            improved = True
                
                current_centers[i] = best_local_c
            
            # Update radii and sum
            current_radii = compute_max_radii(current_centers)
            current_sum = np.sum(current_radii)
            
            if not improved and step > 10:
                # Maybe stuck?
                pass
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()
            
    return best_centers, best_radii, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers, radii, s = optimize_packing()
    
    # Final validation check (optional but good for debugging)
    # We trust the LP constraints, but let's ensure non-negativity
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, s

# Helper to make sure functions are top level
# (They are already defined)

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(r)}, Max radius: {np.max(r)}")
    # Basic validation
    # import numpy as np
    # def validate... (from prompt)
    # print(validate_packing(c, r))