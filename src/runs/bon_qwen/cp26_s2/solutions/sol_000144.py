# sol_000144 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 24d569ae) state=1c999d22 sum of radii=2.576243 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # Helper function to compute the objective and penalty
    # Variables vector: [x1, y1, r1, x2, y2, r2, ...]
    # Shape: (3 * n,)
    def compute_loss(vars_vec, penalty_weight=1000.0):
        c_x = vars_vec[0::3]
        c_y = vars_vec[1::3]
        r = vars_vec[2::3]
        
        # Objective: Minimize negative sum of radii
        loss = -np.sum(r)
        
        # Penalty for boundary violations
        # x - r >= 0  =>  r - x <= 0
        # 1 - x - r >= 0 => x + r - 1 <= 0
        # y - r >= 0
        # 1 - y - r >= 0
        
        # We want to penalize if constraints are violated.
        # Constraint: g(v) >= 0. Penalty if g(v) < 0.
        # Here constraints are:
        # 1. x_i - r_i >= 0
        # 2. 1 - x_i - r_i >= 0
        # 3. y_i - r_i >= 0
        # 4. 1 - y_i - r_i >= 0
        
        # Violation amount: max(0, -g(v))
        # Penalty: weight * sum(violation^2)
        
        # Boundary penalties
        # x_i < r_i  =>  r_i - x_i > 0
        # 1 - x_i < r_i => x_i + r_i - 1 > 0
        
        # Let's define violation magnitudes
        viol_boundary = np.maximum(0, r - c_x) + \
                        np.maximum(0, c_x + r - 1.0) + \
                        np.maximum(0, r - c_y) + \
                        np.maximum(0, c_y + r - 1.0)
        
        loss += penalty_weight * np.sum(viol_boundary ** 2)
        
        # Pairwise overlap penalties
        # Constraint: d_ij^2 - (r_i + r_j)^2 >= 0
        # Violation if d_ij < r_i + r_j
        # Magnitude: (r_i + r_j - d_ij)
        
        # Vectorized computation for performance
        # d_x matrix
        dx = c_x[:, np.newaxis] - c_x[np.newaxis, :]
        dy = c_y[:, np.newaxis] - c_y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        # Add a small epsilon to sqrt to avoid division by zero issues in gradients if needed,
        # though here we just need distance for penalty.
        # Actually sqrt is fine, but let's be careful.
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap amount: positive if overlapping
        overlap = np.maximum(0.0, r_sum - dist)
        
        # Sum of squares of overlaps
        # Only need upper triangle to avoid double counting, 
        # but summing all and dividing by 2 is easier or just taking upper triangle.
        # Taking upper triangle is safer.
        triu_idx = np.triu_indices(n, k=1)
        overlap_sum_sq = np.sum(overlap[triu_idx] ** 2)
        
        loss += penalty_weight * overlap_sum_sq
        
        return loss

    def get_gradient(vars_vec, penalty_weight=1000.0):
        # Numerical gradient is slow, but L-BFGS-B approximates Hessian.
        # Providing gradient speeds it up significantly.
        # Let's implement analytical gradient for the penalty terms.
        
        c_x = vars_vec[0::3]
        c_y = vars_vec[1::3]
        r = vars_vec[2::3]
        
        grad = np.zeros_like(vars_vec)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Boundary terms
        # Term: (r - x)^2 if r > x
        # d/dx: 2(r-x)*(-1) = -2(r-x)
        # d/dr: 2(r-x)*(1) = 2(r-x)
        
        # Term: (x + r - 1)^2 if x + r > 1
        # d/dx: 2(x+r-1)*(1)
        # d/dr: 2(x+r-1)*(1)
        
        # Term: (r - y)^2 if r > y
        # d/dy: -2(r-y)
        # d/dr: 2(r-y)
        
        # Term: (y + r - 1)^2 if y + r > 1
        # d/dy: 2(y+r-1)
        # d/dr: 2(y+r-1)
        
        mask_x_lt_r = r > c_x
        mask_x_gt_1_r = c_x + r > 1.0
        mask_y_lt_r = r > c_y
        mask_y_gt_1_r = c_y + r > 1.0
        
        # Contribution to grad[x] (index 0::3)
        grad[0::3] += penalty_weight * (
            -2.0 * (r - c_x) * mask_x_lt_r + \
             2.0 * (c_x + r - 1.0) * mask_x_gt_1_r
        )
        
        # Contribution to grad[y] (index 1::3)
        grad[1::3] += penalty_weight * (
            -2.0 * (r - c_y) * mask_y_lt_r + \
             2.0 * (c_y + r - 1.0) * mask_y_gt_1_r
        )
        
        # Contribution to grad[r] (index 2::3)
        grad[2::3] += penalty_weight * (
             2.0 * (r - c_x) * mask_x_lt_r + \
             2.0 * (c_x + r - 1.0) * mask_x_gt_1_r + \
             2.0 * (r - c_y) * mask_y_lt_r + \
             2.0 * (c_y + r - 1.0) * mask_y_gt_1_r
        )
        
        # Pairwise terms
        # Term: (r_i + r_j - d_ij)^2 if r_i + r_j > d_ij
        # d/d r_i: 2(r_i + r_j - d_ij) * (1 - (-1/d_ij * (x_i-x_j)/d_ij * (x_i-x_j)/d_ij ... ?))
        # Wait, d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # d/d x_i (r_i + r_j - d_ij) = - (x_i - x_j) / d_ij
        # d/d r_i (r_i + r_j - d_ij) = 1
        
        # So derivative of (r_i + r_j - d_ij)^2 wrt x_i is:
        # 2(r_i + r_j - d_ij) * (-(x_i - x_j)/d_ij)
        
        # We need to sum over all j != i
        
        # Vectorized pairwise gradient computation
        # This is tricky to vectorize cleanly for gradients, but we can do it.
        
        # Overlap matrix (n x n)
        dx = c_x[:, np.newaxis] - c_x[np.newaxis, :]
        dy = c_y[:, np.newaxis] - c_y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = np.maximum(0.0, r_sum - dist)
        
        # We only care about upper triangle (i < j)
        # But for gradient, we can sum over all j and divide by 2?
        # No, derivative wrt x_i involves sum over j != i.
        # The term (r_i + r_j - d_ij)^2 appears once for pair {i,j}.
        # So we sum over j != i.
        
        # Let's compute forces (gradients wrt position)
        # F_x_i = sum_j 2(overlap_ij) * (-(x_i - x_j)/dist_ij)
        # Note: overlap_ij is symmetric. dist_ij symmetric. (x_i - x_j) antisymmetric.
        # So sum over j != i works.
        
        # Create a mask for overlap > 0
        is_overlap = overlap > 1e-9
        
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-9, dist, 1e-9)
        
        # Vector: (x_i - x_j) / dist_ij
        dir_x = dx / safe_dist
        dir_y = dy / safe_dist
        
        # Gradient contribution to x_i from pair (i,j)
        # 2 * (r_i + r_j - d_ij) * (-(x_i - x_j)/d_ij)
        # This is for the term associated with pair (i,j).
        # But in the loss function, we sum over pairs.
        # If we use the full matrix `overlap`, we are double counting if we sum over all j.
        # But derivative of f(x_i, x_j) wrt x_i is what we need.
        # The loss is sum_{i<j} g_{ij}.
        # grad_{x_i} Loss = sum_{j > i} grad_{x_i} g_{ij} + sum_{k < i} grad_{x_i} g_{ki}
        # Since g_{ij} = g_{ji}, this is sum_{j != i} (1/2 * grad_{x_i} g_{ij} + 1/2 * grad_{x_i} g_{ij}?)
        # No. g_{ij} depends on x_i and x_j.
        # grad_{x_i} (sum_{pairs} g) = sum_{j != i} grad_{x_i} g_{ij} (where we treat g_{ij} as the term for pair {i,j})
        # But if we define G matrix where G_ij = g_{ij} for i<j and 0 otherwise?
        # Easier: compute force from j on i.
        # Force on i due to j is -grad_{x_i} g_{ij}.
        # Total force = sum_j Force_{ij}.
        
        # Let's compute the factor: 2 * overlap * (-1/dist)
        # But wait, g = (r_sum - dist)^2.
        # dg/dx_i = 2(r_sum - dist) * (- (x_i - x_j)/dist)
        
        # Let's compute a matrix `grad_x_contrib` where entry (i, j) is the derivative of pair term {i,j} wrt x_i.
        # Actually, let's just compute the sum.
        
        # term = 2 * overlap * (-dir_x)
        # But we must handle the symmetry correctly.
        # The loss includes each pair once.
        # So for variable x_i, we sum contributions from all pairs involving i.
        # Pair {i, j} contributes 2 * overlap_ij * (-dir_x_ij).
        # Since we iterate all j != i, and pair {i, j} is same as {j, i},
        # we just need to be careful not to double count.
        # But if we sum over all j != i, we are considering pair {i, j} from i's perspective.
        # That is correct. Each pair {i, j} has a unique term in the loss.
        # That term contributes to grad x_i and grad x_j.
        # So we just compute the contribution of pair {i, j} to grad x_i.
        # Which is 2 * overlap_ij * (-dir_x_ij).
        # And sum over all j != i.
        
        # Wait, overlap matrix is symmetric. dir_x is antisymmetric.
        # If we sum over all j, we get correct gradient.
        
        # However, overlap matrix includes diagonal (0) and full triangle.
        # We should zero out diagonal and lower/upper triangle?
        # Actually, if we sum over all j != i, we count each pair twice?
        # No. The loss is sum_{i<j} L_{ij}.
        # d/dx_i (L_{ij}) is non-zero.
        # d/dx_i (L_{ki}) where k < i is also non-zero.
        # So total derivative is sum_{j>i} dL_{ij}/dx_i + sum_{k<i} dL_{ki}/dx_i.
        # Since L_{ij} = L_{ji}, this is sum_{j != i} dL_{ij}/dx_i.
        # But we must use the same L_{ij} value.
        # So yes, sum over all j != i of the derivative of the pair term.
        
        # Derivative of (r_i + r_j - d_ij)^2 wrt x_i:
        # 2(r_i + r_j - d_ij) * (-(x_i - x_j)/d_ij)
        
        # Let's compute this matrix.
        # val = 2 * (r_sum - dist) * (-dir_x)
        # But only where overlap > 0.
        
        term = 2.0 * (r_sum - dist) * (-dir_x)
        term = np.where(is_overlap, term, 0.0)
        
        # Sum over columns (j) for each row (i), excluding diagonal
        grad_x_pairs = np.sum(term, axis=1)
        
        # But wait, in the matrix `term`, entry (i, j) corresponds to pair {i, j}.
        # Entry (j, i) also corresponds to pair {j, i} which is same pair.
        # If we sum over all j, we are adding the contribution of pair {i, j} twice?
        # No.
        # The term in loss is L_{ij} (for i<j).
        # d/dx_i L_{ij} depends on x_i.
        # d/dx_i L_{ji} (where j>i) ... wait, L is defined for unordered pairs.
        # If we represent L as a matrix M where M_ij = L_{ij} for i<j, M_ji = 0?
        # No, easier to think: Loss = 0.5 * sum_{i != j} L_{ij}.
        # Then d/dx_i Loss = 0.5 * sum_{j != i} d/dx_i L_{ij}.
        # But L_{ij} in my formula used (r_i+r_j-dist).
        # If we sum over all j != i, we are effectively doing 0.5 * sum_{j != i} ... if we use the full symmetric matrix?
        # Let's check.
        # Loss = sum_{i<j} f(x_i, x_j).
        # d/dx_i Loss = sum_{j>i} df/dx_i + sum_{k<i} df/dx_i (where f is f(x_k, x_i)).
        # Since f(x_k, x_i) = f(x_i, x_k) and df/dx_i(x_k, x_i) = -df/dx_k(x_k, x_i) ... no.
        # f(x, y) depends on x and y.
        # df/dx (x, y) is derivative wrt first arg.
        # df/dy (x, y) is derivative wrt second arg.
        # Here x and y are symmetric in distance.
        # f(x_i, x_j) = g(||x_i - x_j||).
        # d/dx_i f(x_i, x_j) = g' * (x_i - x_j)/d.
        # d/dx_j f(x_i, x_j) = g' * (x_j - x_i)/d = - d/dx_i f(x_i, x_j).
        # So, d/dx_i (sum_{i<j} f_{ij}) = sum_{j>i} d/dx_i f_{ij} + sum_{k<i} d/dx_i f_{ki}.
        # Note f_{ki} is same function as f_{ik}.
        # d/dx_i f_{ki} (derivative wrt second argument of f_{ki})
        # = d/dy f(x_k, y) |_{y=x_i}.
        # Which is equal to - d/dx f(x_k, x_i) (derivative wrt first arg).
        # So sum_{k<i} d/dx_i f_{ki} = - sum_{k<i} d/dx_k f_{ki} ? No.
        # It is sum_{k<i} [ derivative of f(x_k, x_i) wrt x_i ].
        # Which is sum_{k<i} [ - derivative of f(x_k, x_i) wrt x_k ].
        # This doesn't help simplify to a single matrix sum easily without care.
        
        # Simplest way:
        # Compute full matrix of derivatives D_ij = d/dx_i (PairTerm_ij)
        # PairTerm_ij = (r_i + r_j - d_ij)^2
        # D_ij = 2(r_i+r_j-d_ij) * (-(x_i-x_j)/d_ij)
        # This D_ij is computed for all i, j.
        # Note D_ij = - D_ji?
        # Let's check.
        # D_ji = 2(r_j+r_i-d_ji) * (-(x_j-x_i)/d_ji)
        # d_ji = d_ij.
        # -(x_j - x_i) = (x_i - x_j).
        # So D_ji = 2(r_sum - d) * (x_i - x_j)/d = - D_ij.
        # Yes, antisymmetric.
        # The contribution to gradient of x_i from pair {i, j} is:
        # If i < j: term is L_{ij}. Derivative wrt x_i is D_ij.
        # If i > j: term is L_{ji}. Derivative wrt x_i is derivative wrt second arg of L_{ji}.
        # L_{ji}(x_j, x_i). Derivative wrt x_i is D'_{ji} where D' is derivative wrt second arg.
        # D'_{ji} = - D_{ji} (since antisymmetric in args? No, D was derivative wrt first arg).
        # Wait.
        # Let f(u, v) = h(||u-v||).
        # grad_u f = h' * (u-v)/||u-v||.
        # grad_v f = h' * (v-u)/||v-u|| = - grad_u f.
        # So if we compute matrix M where M_ij = grad_{x_i} (PairTerm(x_i, x_j)),
        # Then M_ij = - M_ji.
        # The loss is sum_{i<j} PairTerm(x_i, x_j).
        # grad_{x_i} Loss = sum_{j>i} M_ij + sum_{k<i} (grad_{x_i} PairTerm(x_k, x_i)).
        # grad_{x_i} PairTerm(x_k, x_i) is gradient wrt second argument.
        # Which is - M_ki (since M_ki is grad wrt first arg x_k).
        # So grad_{x_i} Loss = sum_{j>i} M_ij - sum_{k<i} M_ki.
        # Since M is antisymmetric, M_ki = - M_ik.
        # So - sum_{k<i} M_ki = - sum_{k<i} (-M_ik) = sum_{k<i} M_ik.
        # So grad_{x_i} Loss = sum_{j>i} M_ij + sum_{k<i} M_ik = sum_{j != i} M_ij.
        # So we just need to sum the rows of the matrix M (where M_ij is derivative wrt x_i).
        # And M_ij = 2(r_sum - dist) * (-(x_i - x_j)/dist).
        
        # So `grad_x_pairs` computed as sum of rows of `term` is correct?
        # `term` matrix entry (i, j) was 2(r_sum - dist) * (-(x_i - x_j)/dist).
        # This is exactly M_ij.
        # So summing over j gives sum_{j} M_ij.
        # But we must exclude j=i? M_ii = 0 (dist=0 undefined, but handled).
        # And we must check if we double count?
        # No, sum_{j != i} M_ij is the correct gradient.
        # But wait, in `term` matrix, we have entries for all i, j.
        # If we sum all columns for row i, we include M_ij for all j.
        # Is that correct?
        # Yes, because for j < i, the pair is {j, i}, and contribution to x_i is M_ij?
        # Wait.
        # Pair {j, i} with j < i. Term is L_{ji}(x_j, x_i).
        # Gradient wrt x_i is grad_{second_arg} L_{ji}.
        # We established this is - M_ji (where M is grad wrt first arg).
        # And M_ji = - M_ij.
        # So - M_ji = M_ij.
        # So yes, M_ij is the contribution of pair {i, j} to gradient of x_i, regardless of order.
        # So summing M_ij over all j != i works.
        # And M_ij is stored in `term[i, j]`.
        # So `np.sum(term, axis=1)` is correct.
        # We just need to ensure diagonal is 0.
        
        np.fill_diagonal(term, 0)
        grad[0::3] += penalty_weight * np.sum(term, axis=1)
        
        # Gradient wrt y
        term_y = 2.0 * (r_sum - dist) * (-dir_y)
        term_y = np.where(is_overlap, term_y, 0.0)
        np.fill_diagonal(term_y, 0)
        grad[1::3] += penalty_weight * np.sum(term_y, axis=1)
        
        # Gradient wrt r
        # Term: (r_i + r_j - d_ij)^2
        # Derivative wrt r_i: 2(r_i + r_j - d_ij) * 1
        # This is symmetric.
        # Contribution to r_i from pair {i, j} is 2(r_sum - dist).
        # Sum over all j != i.
        # Matrix R_term where entry (i, j) is 2(r_sum - dist).
        # This is symmetric.
        # Sum over j != i of R_term[i, j].
        
        r_grad_matrix = 2.0 * (r_sum - dist)
        r_grad_matrix = np.where(is_overlap, r_grad_matrix, 0.0)
        np.fill_diagonal(r_grad_matrix, 0)
        
        grad[2::3] += penalty_weight * np.sum(r_grad_matrix, axis=1)
        
        return grad

    # Generate initial centers using hexagonal packing logic
    def generate_hex_centers(n):
        centers = []
        # Estimate radius to pack n circles.
        # Area ~ 1. n * pi * r^2 ~ 1 * density.
        # density ~ 0.9.
        # r ~ sqrt(0.9 / (26 * pi)) ~ sqrt(0.011) ~ 0.105.
        # Let's start with r = 0.05 to be safe.
        
        r_init = 0.05
        # Hexagonal spacing
        # Vertical distance between rows
        dy = r_init * math.sqrt(3)
        # Horizontal distance between cols
        dx = 2 * r_init
        
        row = 0
        while len(centers) < n:
            y = r_init + row * dy
            if y + r_init > 1.0:
                break
            
            x_start = r_init
            if row % 2 == 1:
                x_start = r_init + dx / 2
            
            x = x_start
            while x + r_init <= 1.0 and len(centers) < n:
                centers.append([x, y])
                x += dx
            row += 1
            
        # If not enough, add random
        while len(centers) < n:
            centers.append([np.random.rand(), np.random.rand()])
            
        return np.array(centers[:n])

    # Run optimization multiple times with different starts
    best_loss = np.inf
    best_vars = None
    
    # Start 1: Hexagonal
    centers_init = generate_hex_centers(n)
    radii_init = np.full(n, 0.05)
    
    vars_init = np.zeros(3 * n)
    vars_init[0::3] = centers_init[:, 0]
    vars_init[1::3] = centers_init[:, 1]
    vars_init[2::3] = radii_init
    
    bounds = [(0, 1) if k % 3 != 2 else (0, 0.5) for k in range(3 * n)]
    
    # Penalty weight
    lam = 5000.0 
    
    # Try optimization
    res = minimize(compute_loss, vars_init, jac=get_gradient, method='L-BFGS-B', 
                   bounds=bounds, args=(lam,), 
                   options={'maxiter': 2000, 'ftol': 1e-12})
    
    if res.fun < best_loss:
        best_loss = res.fun
        best_vars = res.x
        
    # Try a few random perturbations
    for trial in range(5):
        centers_rnd = np.random.rand(n, 2) * 0.6 + 0.2 # Keep away from edges initially
        radii_rnd = np.full(n, 0.04)
        
        vars_rnd = np.zeros(3 * n)
        vars_rnd[0::3] = centers_rnd[:, 0]
        vars_rnd[1::3] = centers_rnd[:, 1]
        vars_rnd[2::3] = radii_rnd
        
        res = minimize(compute_loss, vars_rnd, jac=get_gradient, method='L-BFGS-B', 
                       bounds=bounds, args=(lam,), 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_vars = res.x

    # Extract results
    c_x = best_vars[0::3]
    c_y = best_vars[1::3]
    r = best_vars[2::3]
    
    centers = np.column_stack((c_x, c_y))
    
    # Post-processing: Ensure strict validity
    # 1. Clip radii and centers to boundaries
    for i in range(n):
        # Max possible radius given position
        max_r_x = min(c_x[i], 1.0 - c_x[i])
        max_r_y = min(c_y[i], 1.0 - c_y[i])
        max_r = min(max_r_x, max_r_y)
        r[i] = min(r[i], max_r)
        # If radius reduced, center is still valid?
        # Yes, because r <= dist to boundary implies r <= x and r <= 1-x.
        # Wait, if we reduce r, it's fine.
        # But if we reduced r, maybe we can move center?
        # For now, just clipping is safe.
        
        # Also ensure non-negative
        if r[i] < 0: r[i] = 0
        
    # 2. Resolve overlaps by shrinking radii
    # Iteratively reduce radii of overlapping pairs
    max_iter = 100
    for _ in range(max_iter):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = c_x[i] - c_x[j]
                dy = c_y[i] - c_y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    # Overlap
                    # Reduce both radii equally? Or proportional?
                    # Just reduce sum to match distance
                    excess = r[i] + r[j] - dist
                    r[i] -= excess / 2
                    r[j] -= excess / 2
                    overlap_found = True
                    # Ensure non-negative
                    if r[i] < 0: r[i] = 0
                    if r[j] < 0: r[j] = 0
        if not overlap_found:
            break
            
    # Re-check boundary constraints after shrinking radii (centers might be bad?)
    # Actually shrinking radii never violates boundary if it didn't before.
    # But we clipped radii to boundary earlier.
    # However, moving centers during optimization might have put them near boundary.
    # We should re-clip radii against boundaries again just in case.
    for i in range(n):
        max_r = min(c_x[i], 1.0 - c_x[i], c_y[i], 1.0 - c_y[i])
        if r[i] > max_r + 1e-12:
            r[i] = max_r
            
    # Final validation check (optional but good for debugging)
    # validate_packing(centers, r)
    
    sum_radii = np.sum(r)
    
    return centers, r, sum_radii

if __name__ == "__main__":
    # To test locally
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)
