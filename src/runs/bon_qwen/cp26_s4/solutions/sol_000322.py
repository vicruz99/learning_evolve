# sol_000322 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f00b2e18) state=1e6d76da sum of radii=2.541321 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Strategy: 25 circles in a 5x5 grid, 1 circle in the central hole.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Base radius for the 5x5 grid. 
    # We use 0.1 for the grid circles.
    R = 0.1
    
    # Radius for the 26th circle in the hole.
    # The hole is at the center of a square formed by 4 circles.
    # Distance from hole center to grid circle center is R * sqrt(2).
    # Max radius r = R * sqrt(2) - R.
    # We reduce slightly to ensure valid packing with numerical tolerance.
    r = R * (np.sqrt(2) - 1) - 1e-4

    # 1. Place 25 circles in a 5x5 grid
    # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
    grid_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    idx = 0
    for i, x in enumerate(grid_coords):
        for j, y in enumerate(grid_coords):
            centers[idx] = [x, y]
            radii[idx] = R
            idx += 1
            
    # 2. Place the 26th circle in the central hole
    # The grid has a hole at the center of the square of 4 central circles.
    # The central circles are at (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7).
    # Wait, the grid is 0.1, 0.3, 0.5, 0.7, 0.9.
    # The central hole is at (0.5, 0.5) is occupied by a grid circle.
    # We need to pick an internal hole.
    # Let's pick the hole at (0.2, 0.2)? No, (0.2, 0.2) is between (0.1,0.1) and (0.3,0.3).
    # Let's place it at (0.5, 0.5) is not a hole.
    # Let's place it at (0.2, 0.5)? Between (0.1, 0.5) and (0.3, 0.5).
    # Actually, any hole surrounded by 4 circles works.
    # Let's pick the hole at (0.2, 0.2) relative to the bottom-left corner of the grid?
    # Grid centers: x in {0.1, 0.3, 0.5, 0.7, 0.9}.
    # Midpoints are 0.2, 0.4, 0.6, 0.8.
    # Let's place the 26th circle at (0.5, 0.2)? No, (0.5, 0.2) is on a line.
    # Let's place it at (0.2, 0.2).
    # Surrounded by (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3).
    
    centers[25] = [0.2, 0.2]
    radii[25] = r

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Validate the solution locally
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # The validation function is provided in the prompt, but we can't import it here directly 
    # without the full environment. However, the logic is sound.
