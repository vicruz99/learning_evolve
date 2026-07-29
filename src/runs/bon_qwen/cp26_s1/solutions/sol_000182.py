# sol_000182 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5c6e3651) state=aaf540d9 sum of radii=2.166667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # Define the structure of rows: 5, 5, 5, 5, 6 circles
    # This is a dense packing configuration.
    rows_counts = [5, 5, 5, 5, 6]
    
    # Generate relative coordinates for a hexagonal lattice
    # Spacing between centers in a row: 2 units (diameter 2, radius 1)
    # Vertical spacing: sqrt(3) units
    # Horizontal shift for staggered rows: 1 unit (radius)
    
    relative_centers = []
    y = 0.0
    shift = 0.0
    
    for count in rows_counts:
        for col in range(count):
            x = col * 2.0 + shift
            relative_centers.append((x, y))
        # Alternate shift for next row
        if shift == 0.0:
            shift = 1.0
        else:
            shift = 0.0
        y += np.sqrt(3)
        
    rel_arr = np.array(relative_centers)
    
    # Calculate bounding box of the circles in relative coordinates
    # Radius in relative coordinates is 1.0
    r_rel = 1.0
    min_x = np.min(rel_arr[:, 0]) - r_rel
    max_x = np.max(rel_arr[:, 0]) + r_rel
    min_y = np.min(rel_arr[:, 1]) - r_rel
    max_y = np.max(rel_arr[:, 1]) + r_rel
    
    width_rel = max_x - min_x
    height_rel = max_y - min_y
    
    # Scale to fit in unit square [0, 1] x [0, 1]
    # We want to fit the bounding box into [0, 1]
    scale = 1.0 / max(width_rel, height_rel)
    
    # Apply scale and translate to center
    # New coordinates = (old - min) * scale + offset
    # Actually, to fit in [0,1], we can just map min->0, max->1?
    # No, we have slack if aspect ratio != 1.
    # Better to center the scaled box in [0,1].
    
    # Scaled size
    w_scaled = width_rel * scale
    h_scaled = height_rel * scale
    
    # Offsets to center in [0,1]
    offset_x = (1.0 - w_scaled) / 2.0
    offset_y = (1.0 - h_scaled) / 2.0
    
    # Transform relative centers to absolute centers
    # abs_coord = (rel_coord - min_rel) * scale + offset
    # But rel_coord includes center. We want to transform centers.
    # Center transformation:
    # new_center = (old_center - (min_rel + r_rel? No))
    # Let's work with centers directly.
    # Bounding box of centers is [min_x+r_rel, max_x-r_rel] ? No.
    # min_x was min_center - r_rel. So min_center = min_x + r_rel.
    # Let's just shift centers by -min_x - r_rel? No.
    
    # Simplest:
    # Shift centers so min_x (of circles) is at 0.
    # Circle extent starts at center_x - r_rel.
    # So center_x_start = min_x + r_rel.
    # We want circle extent to start at offset_x.
    # So new_center_x = (old_center_x - center_x_start) * scale + offset_x + (r_scaled)?
    # Let's do it step by step.
    
    # 1. Shift centers so that the packing fits in [0, 1] with some margin.
    # Current centers range: [min_center_x, max_center_x]
    # min_center_x = np.min(rel_arr[:, 0])
    # max_center_x = np.max(rel_arr[:, 0])
    # Circle bounds: [min_center_x - 1, max_center_x + 1]
    
    # We want to map [min_center_x - 1, max_center_x + 1] to [offset_x, 1 - offset_x]?
    # Actually, we want to map the interval [min_center_x - 1, max_center_x + 1] to [0, 1] scaled by 'scale'.
    # The length is width_rel.
    # Scaled length is width_rel * scale.
    # We place this scaled interval starting at offset_x.
    
    # So, for a point P (center), its circle left edge is P - 1.
    # We want new_left_edge = (P - 1 - (min_center_x - 1)) * scale + offset_x
    #                   = (P - min_center_x) * scale + offset_x
    # Then new_center = new_left_edge + r_scaled
    #                 = (P - min_center_x) * scale + offset_x + r_rel * scale
    
    # Let's verify.
    # If P = min_center_x, new_center = offset_x + r_rel*scale.
    # Left edge = offset_x. Correct.
    # If P = max_center_x, new_center = (max_center_x - min_center_x)*scale + offset_x + r_rel*scale
    #                                 = (width_rel - 2*r_rel)*scale + offset_x + r_rel*scale ?
    # No. max_center_x - min_center_x = width_centers.
    # width_rel = width_centers + 2.
    # So max_center_x - min_center_x = width_rel - 2.
    # new_center = (width_rel - 2)*scale + offset_x + scale
    #            = (width_rel - 1)*scale + offset_x.
    # Right edge = new_center + scale = width_rel*scale + offset_x.
    # Since offset_x = (1 - width_rel*scale)/2, right edge = width_rel*scale + 0.5 - 0.5*width_rel*scale = 0.5 + 0.5*width_rel*scale?
    # Wait.
    # If width_rel * scale = 1 (fits perfectly), offset_x = 0.
    # Right edge = 1. Correct.
    
    # So formula: new_center = (P - min_center_x) * scale + offset_x + r_rel * scale
    
    min_center_x = np.min(rel_arr[:, 0])
    min_center_y = np.min(rel_arr[:, 1])
    
    # Re-calculate scale based on centers?
    # No, scale must account for radius.
    # Width of packing = (max_center_x - min_center_x) + 2*r_rel
    # We computed width_rel earlier as max_x - min_x where max_x = max_center + 1.
    # So width_rel is correct.
    
    scale = 1.0 / max(width_rel, height_rel)
    w_scaled = width_rel * scale
    h_scaled = height_rel * scale
    offset_x = (1.0 - w_scaled) / 2.0
    offset_y = (1.0 - h_scaled) / 2.0
    r_scaled = r_rel * scale
    
    init_centers = np.zeros_like(rel_arr)
    init_centers[:, 0] = (rel_arr[:, 0] - min_center_x) * scale + offset_x + r_scaled
    init_centers[:, 1] = (rel_arr[:, 1] - min_center_y) * scale + offset_y + r_scaled
    
    # Initial radius
    r_init = r_scaled
    
    # Optimization variables: [x0, y0, ..., x25, y25, r]
    x0 = init_centers.flatten()
    initial_guess = np.concatenate([x0, [r_init]])
    
    # Bounds
    bounds = []
    for _ in range(2 * n):
        bounds.append((0.0, 1.0))
    bounds.append((0.0, 1.0))
    
    # Constraints
    cons = []
    idx_r = 2 * n
    
    # Boundary constraints
    for i in range(n):
        idx_x = i * 2
        idx_y = i * 2 + 1
        
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_x] - v[idx_r]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_y] - v[idx_r]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]})
        
    # Overlap constraints
    # To reduce constraints, we can only add them for close pairs?
    # But SLSQP handles N^2 constraints okay for N=26 (325 constraints).
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = i * 2
            idx_yi = i * 2 + 1
            idx_xj = j * 2
            idx_yj = j * 2 + 1
            
            def dist_constraint(v, i=i, j=j):
                xi = v[idx_xi]
                yi = v[idx_yi]
                xj = v[idx_xj]
                yj = v[idx_yj]
                r = v[idx_r]
                # Use squared distance to avoid sqrt if possible, but constraint is linear in r?
                # dist >= 2r => dist^2 >= 4r^2.
                # This is non-convex? No, dist^2 is convex, 4r^2 is convex.
                # dist^2 - 4r^2 >= 0 is not necessarily convex (difference of convex).
                # But linearized or original dist - 2r >= 0 is valid.
                # SLSQP uses gradients.
                d2 = (xi - xj)**2 + (yi - yj)**2
                # If d2 is very small, sqrt is unstable, but constraint handles it.
                # Just return d - 2r
                return np.sqrt(d2) - 2.0 * r
            
            cons.append({'type': 'ineq', 'fun': dist_constraint})

    # Objective: Maximize sum of radii = n * r
    def objective(v):
        return -n * v[idx_r]
    
    # Run optimization
    # Use 'SLSQP'
    try:
        res = opt.minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'ftol': 1e-12, 'maxiter': 500, 'disp': False})
        
        if res.success:
            final_centers = res.x[:2*n].reshape((n, 2))
            final_r = res.x[idx_r]
        else:
            # Fallback
            final_centers = init_centers
            final_r = r_init
    except Exception:
        final_centers = init_centers
        final_r = r_init

    # Ensure radii are equal for the output (since we optimized single r)
    radii = np.full(n, final_r)
    sum_radii = np.sum(radii)
    
    # Final validation/clamping check (optional, but good practice)
    # If any constraint slightly violated due to precision, shrink r slightly.
    # But usually SLSQP is good.
    
    return final_centers, radii, sum_radii
