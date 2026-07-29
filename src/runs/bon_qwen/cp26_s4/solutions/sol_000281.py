# sol_000281 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 20c39dac) state=748b7bd5 sum of radii=2.546742 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize as opt

# Helper function to calculate distance between two points
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Helper function to validate packing (provided in prompt, used for checking)
def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0: return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(centers[i], centers[j])
            if d < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing():
    """
    Finds a packing of 26 circles in a unit square maximizing sum of radii.
    """
    N = 26
    
    # --- Initialization ---
    # Strategy: Start with a dense hexagonal-like grid or random positions.
    # A grid is a good starting point for local optimization.
    # We want to pack 26 circles. A 5x5 grid fits 25. We can start with 25 in grid + 1 random.
    
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.05) # Start with small radii
    
    # Place first 25 in a 5x5 grid
    idx = 0
    for r in range(5):
        for c in range(5):
            centers[idx, 0] = 0.1 + c * 0.2
            centers[idx, 1] = 0.1 + r * 0.2
            idx += 1
    
    # Place 26th circle in a gap (e.g., center of a hole)
    # Holes in 5x5 grid are at (0.2, 0.2), (0.2, 0.4), etc.
    # Let's place it at (0.2, 0.2)
    centers[idx, 0] = 0.2
    centers[idx, 1] = 0.2
    
    # --- Optimization Phase 1: Repulsive Forces & Radius Growth ---
    # We simulate circles growing and repelling each other to find a valid configuration
    # with large radii.
    
    # Learning rate for radius growth
    dr = 0.001
    # Learning rate for movement
    move_lr = 0.01
    
    # Number of iterations for simulation
    num_sim_steps = 2000
    
    # Random seed for reproducibility if needed, but deterministic here
    np.random.seed(42)
    
    for step in range(num_sim_steps):
        # 1. Try to grow radii
        # Grow all radii slightly
        # To avoid immediate collision, grow proportional to current size or uniformly
        # Uniform growth is simpler
        radii += dr
        
        # 2. Resolve overlaps
        # If circles overlap, push them apart.
        # We also need to ensure they stay in bounds.
        
        # Iterate multiple times per step to resolve multiple overlaps
        for _ in range(10):
            changed = False
            # Check pairwise overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist(centers[i], centers[j])
                    sum_r = radii[i] + radii[j]
                    if d < sum_r and d > 1e-9:
                        # Overlap detected
                        overlap = sum_r - d
                        # Move centers apart along the line connecting them
                        # Force proportional to overlap
                        force = overlap * 0.5 * move_lr
                        
                        dx = (centers[i][0] - centers[j][0]) / d
                        dy = (centers[i][1] - centers[j][1]) / d
                        
                        # Move i away from j
                        centers[i][0] += dx * force
                        centers[i][1] += dy * force
                        
                        # Move j away from i
                        centers[j][0] -= dx * force
                        centers[j][1] -= dy * force
                        
                        changed = True
            
            # Check boundary constraints
            for i in range(N):
                r = radii[i]
                # X bounds
                if centers[i][0] < r:
                    centers[i][0] = r
                    changed = True
                elif centers[i][0] > 1.0 - r:
                    centers[i][0] = 1.0 - r
                    changed = True
                # Y bounds
                if centers[i][1] < r:
                    centers[i][1] = r
                    changed = True
                elif centers[i][1] > 1.0 - r:
                    centers[i][1] = 1.0 - r
                    changed = True
            
            if not changed:
                break
        
        # If we can't resolve overlaps without shrinking, shrink radii slightly?
        # Actually, for maximization, we want to keep radii large.
        # If the configuration is stuck (valid), we keep growing.
        # If invalid (still overlaps after resolution attempts), we might need to shrink.
        # But our resolution moves centers. If centers hit walls, radii must shrink.
        
        # Check if any circle is still invalid (inside wall or overlap)
        # Simple check: if any center is pushed out of [0,1] during move (handled by clamp)
        # But if radius is too large for position, clamp handles it by reducing effective radius?
        # No, clamp fixes position. If radius > position, it's invalid.
        # We must ensure radii[i] <= min(x, 1-x, y, 1-y).
        
        valid = True
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            max_r = min(x, 1-x, y, 1-y)
            if r > max_r + 1e-12:
                # Radius too large for position
                radii[i] = max_r
                valid = False # Indicates we had to shrink
        
        # Check overlaps again after boundary fixes
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist(centers[i], centers[j])
                    if d < radii[i] + radii[j] - 1e-9:
                        # Still overlapping, shrink both to satisfy constraint
                        # Equal shrink
                        shrink = (radii[i] + radii[j] - d) * 0.5
                        radii[i] -= shrink
                        radii[j] -= shrink
                        valid = False

        # Reduce growth rate over time to converge
        if step % 500 == 0 and step > 0:
            dr *= 0.5
            move_lr *= 0.5

    # --- Optimization Phase 2: Local Optimization (Scipy) ---
    # Use scipy to fine-tune positions and radii to maximize sum.
    # We fix the radii first, then optimize positions? 
    # Or optimize both. Optimizing both with constraints is hard.
    # Better: Fix radii, optimize positions to minimize overlap penalty.
    # Then fix positions, maximize radii.
    # Iterate.
    
    # Let's try a direct optimization of the objective: sum(radii)
    # with constraints.
    # Variables: 26*3 = 78.
    # Constraints: 26*4 + 26*25/2 ~ 400.
    # SLSQP might be slow.
    
    # Alternative: Iterative improvement.
    # 1. Fix radii, push centers apart (minimize repulsion energy).
    # 2. Fix centers, expand radii as much as possible.
    
    for iteration in range(50):
        # Step A: Fix radii, optimize positions to minimize overlaps
        # Objective: Sum of squared overlaps (penalty)
        # We want overlaps <= 0.
        
        def objective_positions(vars):
            # vars is flattened (N, 2)
            centers_arr = np.resize(vars, (N, 2))
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist(centers_arr[i], centers_arr[j])
                    req = radii[i] + radii[j]
                    if d < req:
                        penalty += (req - d)**2
            # Boundary penalty
            for i in range(N):
                r = radii[i]
                x, y = centers_arr[i]
                if x < r: penalty += (r - x)**2
                if x > 1-r: penalty += (x - (1-r))**2
                if y < r: penalty += (r - y)**2
                if y > 1-r: penalty += (y - (1-r))**2
            return penalty

        # Initial guess for optimizer
        x0 = centers.flatten()
        
        # Minimize penalty
        # Use BFGS or L-BFGS-B
        res = opt.minimize(objective_positions, x0, method='L-BFGS-B', 
                           bounds=[(0, 1)] * (2*N), options={'maxiter': 100})
        centers = np.resize(res.x, (N, 2))
        
        # Step B: Fix centers, maximize radii
        # r_i <= min(boundary_i, min_j(dist_ij - r_j))
        # This is a system of inequalities.
        # We can solve this by relaxation or simply taking min.
        # Since r_i depends on r_j, we iterate.
        
        for _ in range(20):
            new_radii = np.copy(radii)
            for i in range(N):
                # Max radius allowed by boundaries
                x, y = centers[i]
                r_max = min(x, 1-x, y, 1-y)
                
                # Max radius allowed by neighbors
                # r_i <= dist(i, j) - r_j
                for j in range(N):
                    if i == j: continue
                    d = dist(centers[i], centers[j])
                    r_max = min(r_max, d - radii[j])
                
                # We want to increase r_i, but not decrease others unnecessarily?
                # Actually, if we increase r_i, it constrains neighbors.
                # But here we are just finding the feasible max r_i for fixed others?
                # No, radii are variables.
                # A simple greedy update: r_i = min(r_i, r_max) ? No, we want to increase.
                # r_i = max(r_i, r_max)? No, r_max is an upper bound.
                # The constraint is r_i + r_j <= d.
                # So r_i <= d - r_j.
                # So r_i must be <= min_j (d_ij - r_j).
                # Also r_i <= boundary.
                # So r_i <= min(boundary, min_j(d_ij - r_j)).
                # Let R_i* = min(boundary, min_j(d_ij - r_j)).
                # If r_i < R_i*, we can increase r_i.
                # But increasing r_i might violate r_i <= d_ik - r_k for other k?
                # The term min_j(d_ij - r_j) already accounts for all j.
                # So if we set r_i = R_i*, it satisfies all constraints with respect to current r_j.
                # However, if we update r_i, we must check if it reduces the allowable radius for neighbors.
                # But since we iterate, it will converge.
                
                # Wait, if we set r_i = R_i*, we might make r_i very large?
                # R_i* is an upper bound.
                # If current r_i is already <= R_i*, we can potentially increase r_i to R_i*.
                # But R_i* depends on r_j.
                # If we increase r_i, R_j* might decrease.
                # This is like finding the equilibrium.
                # Just setting r_i = R_i* is not correct because R_i* assumes r_j fixed.
                # If we update sequentially, it might work.
                
                # Actually, the "tightest" constraint is r_i + r_j = d_ij.
                # We can just try to expand all circles.
                # But let's stick to: r_i = min(r_i, R_i*)? No.
                # We want to maximize sum.
                # If r_i < R_i*, we can increase r_i.
                # But if we increase r_i, we might violate r_k + r_i <= d_ik.
                # The value R_i* = min_k (d_ik - r_k).
                # So if we set r_i = R_i*, then for all k, r_i + r_k <= d_ik is satisfied.
                # The only issue is that increasing r_i might require decreasing some r_k later?
                # No, if r_i + r_k <= d_ik holds, and we increase r_i, it might violate.
                # But R_i* is calculated using CURRENT r_k.
                # So setting r_i = R_i* satisfies constraints with CURRENT r_k.
                # But it might violate constraints with UPDATED r_k?
                # This suggests a Jacobi iteration might not converge to max sum.
                
                # Better: Just clamp radii to be valid.
                # But we want to maximize.
                # Maybe just keep radii as is if valid, or shrink if invalid?
                # We want to expand.
                
                # Let's try: r_i = min(r_max, r_i + small_step)?
                # No, let's use the computed upper bound.
                # If r_i < r_max, we can potentially grow.
                # But growing might hurt neighbors.
                # However, since we optimize positions in Step A, we have some slack.
                # Let's just ensure validity and maybe expand slightly?
                # Actually, Step A (positions) minimizes overlap. If overlap is 0, radii are valid.
                # So we just need to ensure radii don't grow beyond what positions allow.
                # But we want to increase radii.
                # We can increase radii if positions allow.
                # But positions are fixed in Step B.
                # So max radius for circle i is R_i*.
                # If r_i < R_i*, we can increase r_i.
                # But if we increase r_i, we reduce slack for neighbors.
                # However, the sum of radii might increase.
                # This is a resource allocation problem.
                # For a tree, we can solve exactly. For general graph, it's harder.
                # But simple heuristic: r_i = min(R_i*, r_i + alpha * (R_i* - r_i)).
                # Or just r_i = R_i*?
                # If we set all r_i = R_i* simultaneously, it's inconsistent.
                # But maybe we can just take the min of current r_i and R_i*? No, that shrinks.
                # We want to increase.
                # Maybe: r_i = max(r_i, R_i*)? No, R_i* is upper bound.
                # If r_i > R_i*, we must shrink.
                # If r_i < R_i*, we can grow.
                # But growing r_i reduces R_k* for neighbors.
                
                # Let's just ensure validity for now. The expansion happens in Step A implicitly?
                # No, Step A minimizes overlap for FIXED radii.
                # If radii are too big, overlap > 0.
                # If we shrink radii, overlap might become 0.
                # But we want to maximize radii.
                # So we should NOT shrink unless necessary.
                # If overlap exists, we must shrink.
                # If no overlap, we can try to grow.
                
                # Let's check validity.
                current_r_max = r_max
                for j in range(N):
                    if i == j: continue
                    d = dist(centers[i], centers[j])
                    current_r_max = min(current_r_max, d - radii[j])
                
                # If current radius is greater than allowed, shrink
                if radii[i] > current_r_max:
                    radii[i] = current_r_max
            
            # After pass, radii might be valid.
            # If we had slack (radii[i] < current_r_max), we could grow.
            # But growing requires checking neighbors.
            # Let's just do one pass of shrinking to ensure validity.
            # The growth will come from the fact that in Step A, we might have moved circles apart,
            # creating slack. But radii were fixed during move.
            # Wait, if we moved circles apart, distance increased.
            # So d_ij increased.
            # So allowable radius increased.
            # But we kept radii fixed.
            # So now radii[i] < d_ij - radii[j] (strictly).
            # So we have slack.
            # We should utilize this slack to increase radii.
            
            # Greedy expansion:
            # Increase r_i by min_slack / 2?
            # min_slack_i = min_j (dist_ij - r_i - r_j)
            # If min_slack_i > 0, we can increase r_i.
            # But increasing r_i reduces slack for j.
            # Let's just increase all r_i by a small amount if valid.
            
            # Calculate max possible increase for each
            increases = np.zeros(N)
            for i in range(N):
                slack = 1.0 # Infinity
                # Boundary slack
                x, y = centers[i]
                slack = min(slack, x - radii[i])
                slack = min(slack, (1-x) - radii[i])
                slack = min(slack, y - radii[i])
                slack = min(slack, (1-y) - radii[i])
                
                # Neighbor slack
                for j in range(N):
                    if i == j: continue
                    d = dist(centers[i], centers[j])
                    slack = min(slack, d - radii[i] - radii[j])
                
                increases[i] = slack
            
            # We can increase r_i by delta.
            # But increasing r_i by delta reduces slack for j by delta.
            # If we increase all by delta, we need delta <= min(increases) / 2?
            # Because r_i + r_j increases by 2*delta.
            # So max uniform increase is min(slack_ij) / 2.
            
            min_slack = np.min(increases)
            if min_slack > 1e-12:
                # We can grow
                # Grow by half the slack to be safe
                grow = min_slack * 0.5
                radii += grow
            else:
                # No room to grow uniformly
                # Maybe some circles can grow, some must shrink?
                # For now, do nothing.
                pass

    # Final validation and correction
    # Ensure strict validity
    # If any overlap, shrink radii slightly
    for _ in range(10):
        for i in range(N):
            for j in range(i+1, N):
                d = dist(centers[i], centers[j])
                if d < radii[i] + radii[j]:
                    shrink = (radii[i] + radii[j] - d) * 0.51 # slightly more to be safe
                    radii[i] -= shrink
                    radii[j] -= shrink
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            max_r = min(x, 1-x, y, 1-y)
            if r > max_r:
                radii[i] = max_r

    # Calculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
