# sol_000131 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5da4630c) state=d6dc37fc sum of radii=2.371251 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Phase 1: Hexagonal Lattice Initialization
    # We create a hexagonal grid pattern and select points that fit within the square.
    # Hexagonal packing is denser, allowing for larger initial radii.
    centers = []
    r_approx = 0.101  # Target radius
    
    # Grid parameters
    dx = 2 * r_approx
    dy = np.sqrt(3) * r_approx
    
    # Generate points in a hexagonal pattern
    # We try to fit roughly 5 rows
    for i in range(6): # Rows
        for j in range(6): # Cols
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            # Normalize to fit roughly in [0, 1]
            # We will center and scale later, but for now just collect valid candidates
            if x <= 1.0 and y <= 1.0:
                centers.append([x, y])
        
    # If we don't have enough points, add more or adjust. 
    # For 26, a 6x6 subset is plenty.
    centers = np.array(centers[:n])
    
    # Normalize centers to fit tightly within [0, 1]x[0, 1]
    # Find bounding box
    min_c = np.min(centers, axis=0)
    max_c = np.max(centers, axis=0)
    # Scale and translate to fit in [0.05, 0.95] initially to give room
    # Actually, let's just scale to fill the square
    # To maximize radii, we want centers to be distributed as wide as possible
    
    # Simple scaling to [0,1]
    # But we need to maintain relative structure.
    # Let's map min_c to 0 and max_c to 1
    scale = 1.0 / (max_c - min_c)
    centers = (centers - min_c) * scale
    # Centers are now in [0,1]x[0,1] but touching boundaries.
    # We want them slightly inside.
    centers = centers * 0.95 + 0.025 # Shift slightly inward
    
    radii = np.full(n, 0.01) # Start with small radii

    # Phase 2: Optimization via Repulsive Forces
    # We will simulate forces: circles repel each other and push against walls to expand.
    
    num_steps = 2000
    lr = 0.01 # Learning rate for position updates
    rad_growth = 0.0005 # How fast radii grow
    
    # To make optimization faster and more robust, we can use a simple gradient ascent on sum of radii
    # with penalty for overlaps.
    # But the force-based approach is easier to implement without complex libraries.
    
    # Let's try a simple iterative expansion and relaxation
    # 1. Increase radii
    # 2. Resolve overlaps by moving centers apart
    
    # Better approach: Coordinate Ascent
    # Fix radii, optimize positions (maximize min distance)
    # Fix positions, optimize radii (min distance to neighbors)
    
    # Let's use scipy to optimize the sum of radii directly with penalties.
    # Objective: Maximize sum(r)
    # Constraints: x,y in [r, 1-r], dist >= r_i + r_j
    
    # We can parameterize by centers and a single variable 'r' if we assume equal radii,
    # but we want unequal.
    
    # Let's define a function that computes the feasible radii for a given set of centers.
    # r_i = min( dist(i,j)/2, x_i, 1-x_i, y_i, 1-y_i )
    # Then we maximize sum(r_i) over centers.
    
    def objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        # Calculate max possible radius for each circle given others
        # r_i = min( dist(i,j)/2 for j!=i, x_i, 1-x_i, y_i, 1-y_i )
        
        # Boundary constraints
        r_max = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1 - centers[:, 1]))
        
        # Inter-circle constraints
        # Vectorized distance calculation
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # dists = dists / 2.0
        
        # r_i = min(r_max_i, min_j(dists[i,j]))
        # dists[i,i] is 0, so we need to ignore diagonal.
        
        # Efficient min calculation
        # dists is (n, n)
        # We want min over j!=i
        
        # Let's compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = dists / 2.0
        
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        
        min_dists = np.min(dists, axis=1)
        
        radii = np.minimum(r_max, min_dists)
        
        return -np.sum(radii) # Negate for minimization

    # Use a local optimizer starting from our hex grid
    # Nelder-Mead or Powell might be good for non-smooth objectives
    # But the objective is non-smooth (min function).
    # However, it is Lipschitz continuous.
    
    # Initial guess
    x0 = centers.flatten()
    
    # We can run a few random restarts or just one good run
    # Powell method is derivative-free and handles non-smoothness reasonably well
    
    # To improve performance, we can run a few iterations of "expansion" manually
    # to get close to the optimum, then use scipy.
    
    # Manual expansion loop
    for step in range(500):
        # Compute radii
        r_max = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1 - centers[:, 1]))
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        radii = np.minimum(r_max, min_dists)
        
        # Identify colliding pairs (where dist < r_i + r_j)
        # Actually, if radii are computed this way, there are no overlaps by definition?
        # Wait, r_i is constrained by r_j via dist/2.
        # So r_i <= dist(i,j)/2 => 2*r_i <= dist => dist >= 2*r_i.
        # But we need dist >= r_i + r_j.
        # The condition r_i <= dist(i,j)/2 ensures r_i + r_j <= dist(i,j) + r_j? No.
        # r_i <= dist/2 and r_j <= dist/2 implies r_i + r_j <= dist.
        # So yes, calculating radii this way guarantees no overlap.
        
        # However, this function defines the "capacity" of the configuration.
        # We want to move centers to increase this capacity.
        
        # Simple heuristic: Move centers slightly in random directions or gradient?
        # Gradient of sum(r_i) w.r.t centers?
        # r_i depends on min of several terms.
        # The active constraint determines the gradient.
        
        # Let's just use scipy with a smooth approximation or just run Powell.
        pass

    # Let's use scipy.optimize.minimize with Powell
    # To avoid getting stuck in bad local minima, we can use multiple starting points
    # But we only have one good start (hex grid).
    
    # The objective function is not differentiable everywhere, but Powell doesn't need derivatives.
    
    result = opt.minimize(objective, x0, method='Powell', 
                          options={'maxiter': 5000, 'ftol': 1e-8, 'xtol': 1e-8})
    
    opt_centers = result.x.reshape(-1, 2)
    
    # Calculate final radii
    r_max = np.minimum(np.minimum(opt_centers[:, 0], 1 - opt_centers[:, 0]),
                       np.minimum(opt_centers[:, 1], 1 - opt_centers[:, 1]))
    
    diff = opt_centers[:, np.newaxis, :] - opt_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    opt_radii = np.minimum(r_max, min_dists)
    
    sum_radii = np.sum(opt_radii)
    
    return opt_centers, opt_radii, sum_radii
