import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # --- 1. Initialization: Hexagonal Packing ---
    # We try to fit 26 circles in a hexagonal pattern.
    # A 5x5 grid holds 25 circles. A hexagonal pattern allows tighter packing.
    # Let's try to determine row lengths.
    # Rows with 5 and 6 circles.
    # Pattern: 5, 6, 5, 6, 4 (Sum = 26)
    # Or 6, 5, 6, 5, 4?
    # Let's try to generate coordinates for a hexagonal lattice.
    
    # Initial guess parameters
    # Estimate radius for 26 circles. 
    # Area approx 1. Density ~0.9. 26 * pi * r^2 = 0.9 => r ~ 0.105.
    # Let's start with r = 0.10
    
    initial_r = 0.10
    
    # Generate centers
    centers = []
    r_est = initial_r
    row_height = r_est * math.sqrt(3)
    
    # Try to fit rows. 
    # If row has k circles, width needed is approx 2*k*r (if touching) but in hex lattice 
    # horizontal spacing is 2r.
    # Actually, for a row of k circles, width is (2k-1)r if touching boundaries?
    # Let's just place them in a grid first and let optimizer fix it.
    # A 5x5 grid plus 1 extra.
    
    # Better: Random shuffle of a dense grid might work, but hex is better.
    # Let's create a list of target centers based on a hexagonal lattice scaled to fit.
    
    # We need to decide on row counts.
    # 5, 6, 5, 6, 4 is a plausible distribution for 26.
    # Total rows = 5.
    # Heights: 5 rows. Vertical span ~ (5-1)*r*sqrt(3) + 2r.
    # If r=0.1, span ~ 4*0.1732 + 0.2 = 0.89 + 0.2 = 1.09. A bit high.
    # Maybe r needs to be smaller initially, or rows fewer?
    # 4 rows? 6, 7, 6, 7 = 26.
    # Vertical span: 3*0.1732 + 0.2 = 0.71 + 0.2 = 0.91. Fits easily.
    # Width for 7 circles: 2*7*r = 1.4r. If r=0.1, width 0.14? No.
    # Width for k circles in a row: centers at r, 3r, ..., (2k-1)r.
    # Extent: 0 to 2kr.
    # For k=7, width 14r. If r=0.1, width 1.4 > 1.
    # So 7 circles in a row is too wide for r=0.1.
    # Max circles in row for r=0.1 is 5 (width 1.0).
    # So we are limited to 5 circles per row if r~0.1.
    # To fit 26 circles with max 5 per row, we need 6 rows.
    # 5, 5, 5, 5, 5, 1?
    # Vertical span for 6 rows: 5*r*sqrt(3) + 2r ~ 5*0.1732 + 0.2 = 1.066.
    # Slightly too tall.
    # This suggests r must be slightly less than 0.1 for a standard grid/hex alignment,
    # OR the circles are not aligned in rows, OR radii are unequal.
    
    # Let's start with a configuration that is valid and dense.
    # A 5x5 grid of radius 0.1 is valid (sum=2.5).
    # We can add the 26th circle in a gap with small radius.
    # This is a valid starting point.
    
    centers = []
    # 5x5 Grid
    for i in range(5):
        for j in range(5):
            x = 0.2 * (i + 0.5) # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
            y = 0.2 * (j + 0.5)
            centers.append([x, y])
            
    # 26th circle: place it in the center? 
    # Center is (0.5, 0.5). Already occupied?
    # The 5x5 grid centers are:
    # (0.1, 0.1), (0.3, 0.1), ...
    # (0.5, 0.5) is occupied (i=2, j=2).
    # Gaps are at centers of squares, e.g., (0.2, 0.2).
    # Distance from (0.2, 0.2) to neighbors (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3)
    # is sqrt(0.1^2 + 0.1^2) = sqrt(0.02) ≈ 0.1414.
    # Radii of neighbors 0.1. Sum 0.1 + r_new <= 0.1414 => r_new <= 0.0414.
    # Let's place it at (0.2, 0.2) with r=0.04.
    centers.append([0.2, 0.2])
    
    # Radii: 0.1 for first 25, 0.04 for last one.
    radii = np.array([0.1] * 25 + [0.04])
    
    # --- 2. Optimization ---
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Order: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i][0]
        x0[3*i+1] = centers[i][1]
        x0[3*i+2] = radii[i]
        
    # Bounds
    # x, y in [0, 1]
    # r >= 0 (and implicitly <= 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Constraints
    # 1. Boundary: x - r >= 0, x + r <= 1 => x >= r, x <= 1-r
    #    y - r >= 0, y + r <= 1 => y >= r, y <= 1-r
    # 2. Non-overlap: dist(i,j)^2 >= (ri + rj)^2
    
    cons = []
    
    # Boundary constraints
    for i in range(n):
        # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: x[3*idx] - x[3*idx+2]})
        # x_i <= 1 - r_i  =>  1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: 1.0 - x[3*idx] - x[3*idx+2]})
        # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: x[3*idx+1] - x[3*idx+2]})
        # y_i <= 1 - r_i
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: 1.0 - x[3*idx+1] - x[3*idx+2]})
        
    # Non-overlap constraints
    # Only add for pairs.
    # For large n, this is many constraints. 26 choose 2 = 325.
    # SLSQP handles this.
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
            
    # Objective: Maximize sum of radii => Minimize negative sum
    def objective(x):
        return -sum(x[3*i+2] for i in range(n))
        
    # Run optimization
    # Method SLSQP
    # Tol should be tight enough
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'ftol': 1e-9, 'maxiter': 500, 'disp': False})
    
    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    valid = True
    if res.success:
        for i in range(n):
            final_centers[i, 0] = res.x[3*i]
            final_centers[i, 1] = res.x[3*i+1]
            final_radii[i] = res.x[3*i+2]
    else:
        # If optimization fails, keep initial guess?
        # Or try a simple fallback
        # Let's just use the initial guess modified slightly if needed, 
        # but usually SLSQP works from a valid start.
        # If it failed, we might have a bad result.
        # Let's fallback to initial config
        final_centers = np.array(centers)
        final_radii = radii

    # Validate
    # Since constraints might be violated slightly due to numerical issues or failure,
    # we should ensure validity.
    # But the problem asks to return the result.
    # Let's do a quick check.
    
    # Check for NaNs
    if np.isnan(final_centers).any() or np.isnan(final_radii).any():
        # Fallback to safe grid
        final_centers = np.array(centers)
        final_radii = radii
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Helper to run and print if executed directly
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Number of circles: {len(r)}")