import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs a hybrid LP and force-based optimization to pack 26 circles in a unit square.
    """
    n_circles = 26
    
    def generate_hexagonal_init(seed=0):
        rng = np.random.default_rng(seed)
        # Hexagonal arrangement parameters
        rows = 5
        # Pattern: 5, 6, 5, 6, 4 circles per row to total 26
        row_counts = [5, 6, 5, 6, 4]
        
        centers = []
        # Estimate dimensions based on target sum ~ 2.6 (avg r ~ 0.1)
        # Spacing roughly 0.2
        x_spacing = 1.0 / 6.0
        y_spacing = (np.sqrt(3) / 2) * x_spacing
        
        y_curr = x_spacing / 2  # Start with some padding
        
        for r_idx, count in enumerate(row_counts):
            # Shift odd rows (0-indexed) by half spacing
            x_start = (x_spacing / 2) if r_idx % 2 == 1 else x_spacing
            
            for i in range(count):
                x = x_start + i * x_spacing
                centers.append([x, y_curr])
            
            y_curr += y_spacing

        # Random perturbation to break symmetry and explore space
        centers = np.array(centers) + rng.normal(0, 0.02, centers.shape)
        
        # Clip to safe interior region to ensure initial LP is feasible
        centers = np.clip(centers, 0.05, 0.95)
        return centers

    def solve_lp_radii(centers):
        n = centers.shape[0]
        x, y = centers[:, 0], centers[:, 1]
        
        # Upper bounds for r_i based on boundaries
        bounds_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Pairwise constraints: r_i + r_j <= d_ij
        # We only need to consider pairs that are "close" to avoid 325 constraints
        # For n=26, 325 is small, but optimization is faster with fewer constraints.
        # Threshold: if distance > 0.8, they can't touch given max radius ~ 0.2
        idx_pairs = []
        d_matrix = np.sqrt(((centers[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
        
        # Mask for diagonal and very far pairs
        mask = (d_matrix > 0.8) | np.eye(n, dtype=bool)
        # Get indices of pairs closer than 0.8
        pairs = np.argwhere(mask)
        
        # Constraints matrix A_ub * r <= b_ub
        # Each row has 1 at col i, 1 at col j
        A_rows = []
        b_vals = []
        
        for i, j in pairs:
            A_rows.append([i, j])
            b_vals.append(d_matrix[i, j])
            
        if len(A_rows) == 0:
            # If no pairs are close, radii are just limited by boundaries
            return bounds_r
            
        A_ub = np.zeros((len(A_rows), n))
        for k, (i, j) in enumerate(A_rows):
            A_ub[k, i] = 1
            A_ub[k, j] = 1
            
        b_ub = np.array(b_vals)
        
        # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
        c_obj = -np.ones(n)
        
        # Bounds for variables: 0 <= r_i <= bounds_r[i]
        bounds = [(0, br) for br in bounds_r]
        
        try:
            res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                # Fallback to boundary limits if LP fails (unlikely)
                return np.maximum(0, bounds_r)
        except Exception:
            return np.maximum(0, bounds_r)

    def optimize_trajectory(initial_centers):
        centers = initial_centers.copy()
        step_size = 0.005
        decay = 0.995
        
        for _ in range(150):
            radii = solve_lp_radii(centers)
            
            # Compute forces
            forces = np.zeros_like(centers)
            d_matrix = np.sqrt(((centers[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Pairwise repulsion for "touching" circles
            threshold = 1e-4
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    d = d_matrix[i, j]
                    # If touching or slightly overlapping (due to numerical precision)
                    if d < radii[i] + radii[j] + threshold:
                        if d > 1e-9:
                            u = (centers[i] - centers[j]) / d
                            # Push apart
                            forces[i] += u
                            forces[j] -= u
                        else:
                            # Avoid division by zero, push random direction
                            forces[i] += np.random.randn(2) * 0.1
                            forces[j] -= np.random.randn(2) * 0.1
                            
            # Boundary repulsion
            x, y = centers[:, 0], centers[:, 1]
            for i in range(n_circles):
                r = radii[i]
                if x[i] < r + threshold:
                    forces[i, 0] += 1.0
                if x[i] > 1.0 - r - threshold:
                    forces[i, 0] -= 1.0
                if y[i] < r + threshold:
                    forces[i, 1] += 1.0
                if y[i] > 1.0 - r - threshold:
                    forces[i, 1] -= 1.0

            centers += step_size * forces
            
            # Clip to valid region (slight padding to keep LP feasible)
            centers = np.clip(centers, 0.001, 0.999)
            
            step_size *= decay
            
        return centers

    # Run multiple trajectories with different random seeds to find global optimum
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try 10 different random perturbations of the hexagonal lattice
    for seed in range(10):
        init_c = generate_hexagonal_init(seed=seed)
        opt_c = optimize_trajectory(init_c)
        opt_r = solve_lp_radii(opt_c)
        curr_sum = np.sum(opt_r)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = opt_c
            best_radii = opt_r

    return best_centers, best_radii, best_sum