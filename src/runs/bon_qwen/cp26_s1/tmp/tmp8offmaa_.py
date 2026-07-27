import numpy as np
    from scipy.optimize import linprog, minimize
    from scipy.spatial.distance import pdist, squareform

    def get_max_radius_sum(centers):
        # Solve LP to maximize sum(r_i)
        # Variables: r_0 ... r_25
        # Maximize sum(r) -> Minimize -sum(r)
        n = centers.shape[0]
        c_obj = -np.ones(n)
        
        # Constraints: r_i >= 0 (handled by bounds)
        # r_i <= x_i, r_i <= 1-x_i, etc.
        # r_i + r_j <= dist_ij
        
        # A_ub * r <= b_ub
        
        # Boundary constraints:
        # r_i <= x_i
        # -r_i <= -x_i (wait, r_i is positive)
        # Actually r_i <= x_i is 1*r_i <= x_i.
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            # r_i <= x
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(x)
            # r_i <= 1-x
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1-x)
            # r_i <= y
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(y)
            # r_i <= 1-y
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1-y)
            
        # Overlap constraints: r_i + r_j <= dist
        dists = squareform(pdist(centers))
        for i in range(n):
            for j in range(i+1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None)] * n
        
        # Solve
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun
        else:
            return 0 # Should not happen

    def objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        # Clamp centers to [0,1] to avoid invalid LP? 
        # Actually if centers outside, LP might give large radii?
        # But r_i <= x_i constraint handles it.
        # If x_i < 0, r_i <= negative -> r_i=0.
        # So it's safe.
        return -get_max_radius_sum(centers)

    # Initial guess
    # Grid 5x5 + 1
    # ...