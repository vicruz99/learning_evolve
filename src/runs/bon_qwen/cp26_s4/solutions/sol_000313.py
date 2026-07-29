# sol_000313 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5cd869be) state=edc73138 sum of radii=2.438966 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Target number of circles
    n = 26
    
    # Strategy: 6 rows of circles in a hexagonal lattice pattern.
    # Row sizes: 5, 4, 5, 4, 5, 3 (Total = 26)
    row_sizes = [5, 4, 5, 4, 5, 3]
    num_rows = len(row_sizes)
    
    # Calculate optimal radius based on the vertical height of 6 rows in hexagonal packing
    # Height constraint: 2r + (num_rows - 1) * r * sqrt(3) = 1
    # r * (2 + (num_rows - 1) * sqrt(3)) = 1
    r = 1.0 / (2.0 + (num_rows - 1) * math.sqrt(3))
    
    centers = np.zeros((n, 2))
    radii = np.full(n, r)
    
    current_idx = 0
    # Vertical position of the first row center
    y = r
    
    for i in range(num_rows):
        count = row_sizes[i]
        
        # Determine horizontal offset for hexagonal staggering
        # Even rows (0, 2, 4) start at x = r. 
        # Odd rows (1, 3, 5) start at x = 2r to nest into the gaps of the previous row.
        if i % 2 == 0:
            x_start = r
        else:
            x_start = 2 * r
            
        for j in range(count):
            # Place center
            cx = x_start + j * 2 * r
            cy = y
            centers[current_idx] = [cx, cy]
            current_idx += 1
            
        # Move y to the next row
        y += r * math.sqrt(3)
        
    sum_radii = np.sum(radii)
    return centers, radii, float(sum_radii)

# Example usage for validation (if running locally):
# if __name__ == "__main__":
#     centers, radii, s = run_packing()
#     print(f"Sum of radii: {s}")
