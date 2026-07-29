# sol_000255 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 02c202ea) state=b7a36619 sum of radii=2.438964 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a hexagonal lattice arrangement with row distribution [5, 4, 5, 4, 5, 3].
    """
    
    # Number of circles
    n = 26
    
    # Hexagonal lattice constants
    sqrt_3 = np.sqrt(3)
    
    # Row configuration: [5, 4, 5, 4, 5, 3] circles
    rows = [5, 4, 5, 4, 5, 3]
    num_rows = len(rows)
    
    # We need to find the max radius r such that the pattern fits in [0, 1] x [0, 1].
    # Let's analyze the bounding box of the pattern in terms of r.
    # Horizontal span:
    # Even rows (5 circles) span from x to x + 4*(2r) = x + 8r.
    # Odd rows (4 circles) span from x+r to x+r + 3*(2r) = x + 7r.
    # To minimize width, we align the "center of mass" or just check extremes.
    # If we place the 5-circle rows starting at x=0 (relative), they end at 8r.
    # The 4-circle rows shifted by r start at r and end at 7r + r = 8r.
    # So the horizontal width of the pattern is 8r.
    # Vertical span:
    # 6 rows. Vertical distance between centers is r*sqrt(3).
    # Total height from first row center to last row center is (num_rows - 1) * r * sqrt(3).
    # Adding radius r on top and bottom, total height is 2r + (num_rows - 1) * r * sqrt(3).
    
    # Constraints:
    # 1. Width <= 1 => 8r <= 1 => r <= 1/8 = 0.125
    # 2. Height <= 1 => 2r + 5 * r * sqrt(3) <= 1 => r * (2 + 5*sqrt(3)) <= 1
    #    2 + 5*1.732... = 2 + 8.66... = 10.66...
    #    r <= 1 / 10.66... approx 0.0938
    
    # The height constraint is tighter.
    # r_max = 1 / (2 + 5 * sqrt(3))
    
    # However, we can shift the rows horizontally to optimize or just center them.
    # Let's calculate the exact max r based on the height constraint.
    # Note: The width of 8r is for the span of centers. The circles extend r beyond centers.
    # So if centers span [x_min, x_max], circles span [x_min - r, x_max + r].
    # Width = (x_max - x_min) + 2r.
    # For 5 circles at 0, 2r, 4r, 6r, 8r (centers), span is 8r.
    # Circle extent: -r to 9r. Width 10r? 
    # Wait, if centers are at 0, 2r, 4r, 6r, 8r. 
    # Leftmost edge of first circle: 0 - r = -r.
    # Rightmost edge of last circle: 8r + r = 9r.
    # Total width = 10r.
    # Constraint: 10r <= 1 => r <= 0.1.
    
    # Let's re-evaluate shifted rows.
    # Row 0 (5 circles): centers at 0, 2r, 4r, 6r, 8r. 
    # Edges: [-r, 9r]. Width 10r.
    # Row 1 (4 circles): shifted by r. Centers at r, 3r, 5r, 7r.
    # Edges: [r-r, 7r+r] = [0, 8r]. Width 8r.
    # So Row 1 is narrower and fits inside the x-range of Row 0 if Row 0 is centered?
    # If Row 0 is centered in [0, 1], its range is [0.5 - 5r, 0.5 + 5r].
    # Edges: [0.5 - 6r, 0.5 + 6r]. Width 12r? No.
    # Let's stick to bounding box of centers.
    # Centers of Row 0: x in [0, 8r].
    # Centers of Row 1: x in [r, 7r].
    # Overall center x-range: [0, 8r].
    # Circle x-range: [-r, 9r].
    # To fit in [0, 1], we need to shift the pattern.
    # Let pattern start at x_start.
    # Min x of circles: x_start - r >= 0 => x_start >= r.
    # Max x of circles: x_start + 8r + r <= 1 => x_start + 9r <= 1 => x_start <= 1 - 9r.
    # So we need r <= 1 - 9r => 10r <= 1 => r <= 0.1.
    
    # Now check height.
    # Centers y-range: [0, 5*r*sqrt(3)].
    # Circle y-range: [-r, 5*r*sqrt(3) + r].
    # To fit in [0, 1]:
    # Shift y_start.
    # y_start - r >= 0 => y_start >= r.
    # y_start + 5*r*sqrt(3) + r <= 1 => y_start <= 1 - r - 5*r*sqrt(3).
    # We need r <= 1 - r - 5*r*sqrt(3) => 2r + 5*r*sqrt(3) <= 1.
    # r(2 + 5*sqrt(3)) <= 1.
    # r <= 1 / (2 + 5*sqrt(3)).
    
    # 2 + 5*1.73205 = 10.66025.
    # 1 / 10.66025 approx 0.0938.
    # Since 0.0938 < 0.1, the height constraint is the bottleneck.
    
    # Let's compute r_max.
    r = 1.0 / (2.0 + 5.0 * sqrt_3)
    
    # To be safe against numerical errors and satisfy "inside unit square",
    # we can scale down slightly.
    r = r * 0.999999
    
    # Now generate coordinates.
    # We center the pattern in the square.
    # X-bounds of centers: [0, 8r]. Center at 4r.
    # Shift to center of square (0.5).
    # x_shift = 0.5 - 4r.
    # But we must ensure [x_shift, x_shift + 8r] allows circles to be in [0, 1].
    # Left edge: x_shift - r = 0.5 - 5r.
    # Right edge: x_shift + 8r + r = 0.5 + 5r.
    # Since r approx 0.094, 5r approx 0.47. 
    # 0.5 - 0.47 = 0.03 > 0. 
    # 0.5 + 0.47 = 0.97 < 1.
    # So centering works.
    
    x_shift = 0.5 - 4 * r
    
    # Y-bounds of centers: [0, 5*r*sqrt(3)]. Center at 2.5*r*sqrt(3).
    # Shift to center of square (0.5).
    # y_shift = 0.5 - 2.5 * r * sqrt_3.
    # Check edges:
    # Bottom: y_shift - r = 0.5 - r - 2.5*r*sqrt(3) = 0.5 - r(1 + 2.5*sqrt(3)).
    # 1 + 2.5*1.732 = 1 + 4.33 = 5.33.
    # 5.33 * 0.094 approx 0.5. 
    # It should be very close to 0.
    # Top: y_shift + 5*r*sqrt(3) + r = 0.5 + r(1 + 2.5*sqrt(3)).
    # Close to 1.
    
    y_shift = 0.5 - 2.5 * r * sqrt_3
    
    centers = []
    
    for row_idx, count in enumerate(rows):
        y_center = y_shift + row_idx * r * sqrt_3
        
        # Determine x-offset for this row
        # Even rows (0, 2, 4): start at 0 relative to pattern
        # Odd rows (1, 3, 5): start at r relative to pattern (shifted)
        if row_idx % 2 == 0:
            x_start = x_shift
        else:
            x_start = x_shift + r
            
        for col_idx in range(count):
            x_center = x_start + col_idx * 2 * r
            centers.append([x_center, y_center])
            
    centers = np.array(centers)
    radii = np.full(n, r)
    
    sum_radii = np.sum(radii)
    
    # Validate just to be sure (internal check, not printed)
    # validate_packing(centers, radii) would be called by user
    
    return centers, radii, sum_radii

# Note: The validate_packing function is provided in the prompt and not defined here 
# as per instructions to not modify it, but we assume it exists in the execution context.
# However, the prompt says "We will run the below validation function", implying it's external.
# The code block only needs to define run_packing.

if __name__ == "__main__":
    # Quick local test if run directly, though not required by prompt structure
    # centers, radii, s = run_packing()
    # print(f"Sum of radii: {s}")
    pass
