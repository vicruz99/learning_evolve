# sol_000285 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b0b06613) state=939bb908 sum of radii=2.492573 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses an iterative approach:
    1. Fix centers, solve LP to find optimal radii.
    2. Compute gradient (forces) from LP duals.
    3. Update centers to increase distances (relax constraints).
    """
    
    n = 26
    
    # --- Initialization ---
    # Start with a 5x5 grid + 1 circle, slightly perturbed
    np.random.seed(42)
    centers = np.zeros((n, 2))
    
    # 5x5 grid points
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    
    # Add 26th point in a gap
    grid_pts.append([0.2, 0.2])
    
    # Shuffle to randomize order (optional, but helps symmetry breaking)
    indices = np.random.permutation(n)
    for k in range(n):
        centers[k] = grid_pts[indices[k]]
        
    # Add small random noise
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    # Clip to ensure inside [0,1] initially
    centers = np.clip(centers, 0.01, 0.99)

    # --- Optimization Loop ---
    step_size = 0.01
    max_iter = 200
    
    for iteration in range(max_iter):
        # 1. Compute distance matrix
        # dist[i, j] = Euclidean distance between centers[i] and centers[j]
        dist_matrix = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
        
        # 2. Setup LP
        # Variables: r_0, ..., r_25
        # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
        c_obj = np.ones(n) * -1.0
        
        # Constraints:
        # A_ub @ r <= b_ub
        # We will collect rows for A_ub and values for b_ub
        
        # Map to keep track of which constraint corresponds to which pair/boundary
        # For gradient calculation, we need to know the dual variable for each constraint
        
        constraints_pairs = [] # List of (i, j) for pairwise
        constraints_boundary = [] # List of (i, bound_type) where bound_type in ['x_min', 'x_max', 'y_min', 'y_max']
        
        A_rows = []
        b_vals = []
        
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        # Only for i < j
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Row for r_i + r_j
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_vals.append(dist_matrix[i, j])
                constraints_pairs.append((i, j, len(b_vals) - 1)) # Store index in marginals
        
        # Boundary constraints
        # r_i <= x_i  => r_i <= centers[i, 0]
        # r_i <= 1 - x_i => r_i <= 1 - centers[i, 0]
        # r_i <= y_i
        # r_i <= 1 - y_i
        
        for i in range(n):
            # x_min: r_i <= x_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(centers[i, 0])
            constraints_boundary.append((i, 'x_min', len(b_vals) - 1))
            
            # x_max: r_i <= 1 - x_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(1.0 - centers[i, 0])
            constraints_boundary.append((i, 'x_max', len(b_vals) - 1))
            
            # y_min: r_i <= y_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(centers[i, 1])
            constraints_boundary.append((i, 'y_min', len(b_vals) - 1))
            
            # y_max: r_i <= 1 - y_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(1.0 - centers[i, 1])
            constraints_boundary.append((i, 'y_max', len(b_vals) - 1))

        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        
        # Bounds for radii: r_i >= 0
        bounds = [(0, None) for _ in range(n)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not res.success:
                # If LP fails, break or handle error
                # This might happen if constraints are contradictory (e.g. centers too close)
                # But with r >= 0, it should always be feasible.
                # Unless dist < 0 which is impossible.
                break
                
            radii = res.x
            marginals = res.ineqlin.marginals
            
            # 3. Compute Forces
            forces = np.zeros((n, 2))
            
            # From pairwise constraints
            # If constraint i,j is active (marginal > 0), we want to increase distance.
            # Force on i is proportional to marginal * direction(i->j)
            # Direction i->j is (centers[j] - centers[i]) / dist
            # Wait, if we increase distance, we pull them apart.
            # If r_i + r_j = dist, increasing dist allows larger r.
            # Gradient of dist w.r.t center i is (centers[i] - centers[j]) / dist.
            # Wait, dist = ||c_i - c_j||. grad_{c_i} dist = (c_i - c_j) / dist.
            # Moving c_i in this direction increases distance.
            # So force on i should be in direction (c_i - c_j).
            
            for i, j, idx in constraints_pairs:
                dual = marginals[idx]
                if dual > 1e-6: # Active constraint
                    d = dist_matrix[i, j]
                    if d > 1e-9:
                        vec = centers[i] - centers[j]
                        force_vec = dual * (vec / d)
                        forces[i] += force_vec
                        forces[j] -= force_vec
            
            # From boundary constraints
            # If r_i = x_i (active), moving x_i right increases bound.
            # Gradient of bound (x_i) w.r.t x_i is 1.
            # So force in +x direction.
            # If r_i = 1 - x_i (active), moving x_i left increases bound (since 1-x_i gets larger).
            # Gradient of bound (1-x_i) w.r.t x_i is -1.
            # So force in -x direction.
            
            for i, b_type, idx in constraints_boundary:
                dual = marginals[idx]
                if dual > 1e-6:
                    if b_type == 'x_min':
                        forces[i, 0] += dual
                    elif b_type == 'x_max':
                        forces[i, 0] -= dual
                    elif b_type == 'y_min':
                        forces[i, 1] += dual
                    elif b_type == 'y_max':
                        forces[i, 1] -= dual
            
            # 4. Update Centers
            # Scale forces to ensure stability
            force_norm = np.linalg.norm(forces)
            if force_norm > 0:
                # Normalize and scale by step_size?
                # Or just use raw forces if they are small enough.
                # Duals can be large. Let's dampen.
                centers += step_size * forces
            
            # Clip centers to stay within [0, 1]
            # Although theoretically they shouldn't go out if constraints handled,
            # numerical issues or large steps might push them out.
            # However, clipping might violate boundary constraints logic (r <= x).
            # If we clip x to 0, r must be 0.
            # The LP will handle r=0 if x=0.
            # So clipping is safe.
            centers = np.clip(centers, 0.0, 1.0)
            
            # Decay step size slightly to converge
            step_size *= 0.995
            
        except Exception as e:
            break

    # Final radii extraction (re-run LP to ensure consistency with final centers)
    dist_matrix = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    
    c_obj = np.ones(n) * -1.0
    A_rows = []
    b_vals = []
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(dist_matrix[i, j])
            
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(centers[i, 0]) # r_i <= x_i
        
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(1.0 - centers[i, 0]) # r_i <= 1-x_i
        
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(centers[i, 1]) # r_i <= y_i
        
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(1.0 - centers[i, 1]) # r_i <= 1-y_i

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        radii = res.x
    except:
        # Fallback if LP fails
        radii = np.zeros(n)

    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    # Validate
    try:
        from validate_packing import validate_packing # Assuming validation function is available or copied
        # Since I can't import from outside, I'll just print results
        # But the problem statement says validate_packing is provided.
        # I will assume the environment has it.
        # For local testing, I can define a simple check.
        
        # Simple check
        valid = True
        n = centers.shape[0]
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if r < 0: valid = False
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                valid = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < radii[i] + radii[j] - 1e-9:
                    valid = False
        print(f"Valid: {valid}")
        
    except Exception as e:
        print(e)
