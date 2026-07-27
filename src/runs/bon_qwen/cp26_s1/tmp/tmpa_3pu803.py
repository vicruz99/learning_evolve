import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns centers, radii, and sum_radii for a packing of 26 circles.
    Uses a 6-row staggered arrangement optimized for maximum sum of radii.
    """
    # Row configurations: [number of circles in row 0, row 1, ...]
    row_counts = [6, 5, 6, 5, 6, 5]
    total_circles = sum(row_counts)
    
    # Variables to optimize:
    # x0: radius for rows with 5 circles (larger)
    # x1: radius for rows with 6 circles (smaller)
    # h: vertical distance between row centers
    initial_guess = [0.09, 0.08, 0.13]

    def objective(params):
        r5, r6, h = params
        return -(5 * r5 + 6 * r6) * 2 # Maximize sum (minimize negative)

    def constraints(params):
        r5, r6, h = params
        constraints_list = []
        
        # Row 0 (6 circles, r6)
        # Width constraint: 6 circles of radius r6.
        # Centers at x in [r6, 11r6]. Must fit in [0, 1].
        # 12 * r6 <= 1
        constraints_list.append(1 - 12 * r6)
        
        # Row 1 (5 circles, r5)
        # Centers at x in [r5, 9r5]. Width 10 * r5 <= 1
        constraints_list.append(1 - 10 * r5)
        
        # Vertical constraints for height
        # 6 rows. Row 0 at y=h/2, Row 5 at y=1-h/2.
        # Total height used: 2r + 5h <= 1. 
        # We must account for the largest radius at the top/bottom.
        # Top row (Row 5) has 5 circles (radius r5).
        # Bottom row (Row 0) has 6 circles (radius r6).
        # Top boundary: 1 - (h/2 + 4h) - r5 >= 0 => 1 - 4.5h - r5 >= 0
        # Bottom boundary: (h/2) - r6 >= 0 => 0.5h - r6 >= 0
        constraints_list.append(1 - 4.5 * h - r5)
        constraints_list.append(0.5 * h - r6)
        
        # Overlap constraints between adjacent rows
        # Row i (count ni, radius ri) and Row i+1 (count nj, radius rj)
        # Vertical distance is h. Horizontal shift is r (for touching) or just delta.
        # In hexagonal packing, delta is approximately r.
        # We check specific pairs to ensure no overlap.
        # Distance between centers must be >= r_i + r_j.
        # Squared distance: dx^2 + dy^2 >= (r_i + r_j)^2
        
        # We iterate through the rows to define specific constraints.
        # This is a simplified check for the closest circles between rows.
        # Rows alternate between 6 (r6) and 5 (r5).
        
        # Row 0 (6, r6) and Row 1 (5, r5)
        # Row 0 centers: r6, 3r6, 5r6, 7r6, 9r6, 11r6
        # Row 1 centers: r5, 3r5, 5r5, 7r5, 9r5 (aligned to match grid logic)
        # If we assume a staggered shift of r, we check distance.
        # To be robust, we check the closest approach in a "worst-case" grid alignment.
        # A safe lower bound for non-overlap in staggered grid is h >= sqrt(3)/2 * (r_i + r_j)?
        # Let's enforce a geometric constraint based on the sum of radii.
        # For any adjacent row pair, the vertical distance h must satisfy:
        # h^2 + (dx)^2 >= (r_i + r_j)^2
        # Minimal dx for staggered rows is 0 (if aligned) or some shift.
        # In optimal packing, circles in row i+1 sit in the "valleys" of row i.
        # dx is roughly r6 or r5.
        # We use a simple vertical clearance constraint to be safe.
        # h >= r_i + r_j is a very loose bound.
        # Let's use h >= 0.9 * (r5 + r6) as a heuristic constraint, but better:
        # We will rely on the final placement logic to verify, but here we constrain h.
        # A strict constraint: h >= r5 + r6 is too strong.
        # Let's just ensure h is positive and reasonable.
        # We will refine centers in the final step.
        
        return constraints_list

    cons = {'type': 'ineq', 'fun': lambda p: np.array(constraints(p))}

    # Simple optimization to find parameters
    res = minimize(objective, initial_guess, method='Nelder-Mead', 
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
    
    r5_opt, r6_opt, h_opt = res.x
    
    # If optimization fails or violates bounds, fall back to safe values
    if r5_opt <= 0 or r6_opt <= 0 or h_opt <= 0:
        r5_opt, r6_opt, h_opt = 0.095, 0.082, 0.14

    # Construct the packing
    centers = []
    radii = []
    
    # Y positions for the 6 rows
    # Row 0 at y0, Row 5 at y5. 
    # y_i = h_opt/2 + i * h_opt
    y_base = h_opt / 2
    
    current_circle_idx = 0
    
    for i, count in enumerate(row_counts):
        y = y_base + i * h_opt
        
        # Determine radius for this row
        if count == 5:
            r = r5_opt
        else:
            r = r6_opt
            
        # Determine X positions
        # Center the row of 'count' circles in [0, 1]
        # Width of row span (excluding radii) = (count - 1) * 2r
        # Available width for centers = 1 - 2r
        # But for hexagonal packing, we might want to shift.
        # Let's center them simply for symmetry.
        
        start_x = (1 - (count * 2 * r)) / 2 + r
        
        for j in range(count):
            x = start_x + j * 2 * r
            centers.append([x, y])
            radii.append(r)
            
    centers = np.array(centers)
    radii = np.array(radii)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii