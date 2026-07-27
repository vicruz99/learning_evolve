import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

N = 26

def con_func(z):
    m = 4*N + N*(N-1)//2
    vals = np.empty(m)
    idx = 0
    for i in range(N):
        x, y, r = z[3*i], z[3*i+1], z[3*i+2]
        vals[idx] = x - r
        vals[idx+1] = 1.0 - x - r
        vals[idx+2] = y - r
        vals[idx+3] = 1.0 - y - r
        idx += 4
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
            xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
            dist_sq = (xi-xj)**2 + (yi-yj)**2
            sum_r = ri + rj
            vals[idx] = dist_sq - sum_r**2
            idx += 1
    return vals

def con_jac(z):
    m = 4*N + N*(N-1)//2
    jac = np.zeros((m, 3*N))
    for i in range(N):
        row = 4*i
        idx = 3*i
        jac[row, idx] = 1.0
        jac[row, idx+2] = -1.0
        jac[row+1, idx] = -1.0
        jac[row+1, idx+2] = -1.0
        jac[row+2, idx+1] = 1.0
        jac[row+2, idx+2] = -1.0
        jac[row+3, idx+1] = -1.0
        jac[row+3, idx+2] = -1.0
        
    row = 4*N
    for i in range(N):
        for j in range(i+1, N):
            idx_i = 3*i
            idx_j = 3*j
            xi, yi, ri = z[idx_i], z[idx_i+1], z[idx_i+2]
            xj, yj, rj = z[idx_j], z[idx_j+1], z[idx_j+2]
            sum_r = ri + rj
            
            jac[row, idx_i] = 2.0*(xi-xj)
            jac[row, idx_i+1] = 2.0*(yi-yj)
            jac[row, idx_i+2] = -2.0*sum_r
            jac[row, idx_j] = 2.0*(xj-xi)
            jac[row, idx_j+1] = 2.0*(yj-yi)
            jac[row, idx_j+2] = -2.0*sum_r
            row += 1
    return jac

def obj_func(z):
    return -np.sum(z[2::3])

def obj_grad(z):
    g = np.zeros(3*N)
    g[2::3] = -1.0
    return g

def run_packing():
    best_z = None
    best_sum_r = -1.0
    
    grid_side = 5
    spacing = 1.0 / (grid_side + 1)
    centers = []
    for i in range(grid_side):
        for j in range(grid_side):
            centers.append([spacing * (i + 1), spacing * (j + 1)])
    centers = np.array(centers[:N-1])
    centers = np.vstack([centers, [0.5, 0.5]])
    
    for seed in range(5):
        np.random.seed(seed)
        z0 = np.zeros(3*N)
        pts = centers + np.random.uniform(-0.02, 0.02, (N, 2))
        pts = np.clip(pts, 0.01, 0.99)
        z0[0::3] = pts[:, 0]
        z0[1::3] = pts[:, 1]
        z0[2::3] = 0.01
        
        bnds = Bounds(0.0, 1.0)
        bnds[2::3] = (1e-5, 0.5)
        
        con = NonlinearConstraint(con_func, np.zeros(4*N + N*(N-1)//2), np.inf, con_jac)
        
        res = minimize(obj_func, z0, method='trust-constr', jac=obj_grad, 
                       bounds=bnds, constraints=con, 
                       options={'maxiter': 800, 'verbose': 0})
        
        current_sum = -res.fun
        c = res.x.reshape(-1, 3)[:, :2]
        r = res.x.reshape(-1, 3)[:, 2]
        
        valid = True
        if np.any(r < 0) or np.any(c < -1e-7) or np.any(c > 1+1e-7):
            valid = False
        else:
            for i in range(N):
                if c[i,0]-r[i] < -1e-7 or c[i,0]+r[i] > 1+1e-7 or c[i,1]-r[i] < -1e-7 or c[i,1]+r[i] > 1+1e-7:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                        if dist < r[i] + r[j] - 1e-7:
                            valid = False
                            break
                    if not valid: break
                    
        if valid and current_sum > best_sum_r:
            best_sum_r = current_sum
            best_z = res.x.copy()
            
    if best_z is None:
        best_z = np.zeros(3*N)
        for i in range(N-1):
            best_z[3*i] = 0.1 + 0.2 * (i // 5)
            best_z[3*i+1] = 0.1 + 0.2 * (i % 5)
            best_z[3*i+2] = 0.1
        best_z[3*(N-1)] = 0.5
        best_z[3*(N-1)+1] = 0.5
        best_z[3*(N-1)+2] = 0.1
        
    centers = best_z.reshape(-1, 3)[:, :2]
    radii = best_z.reshape(-1, 3)[:, 2]
    return centers, radii, np.sum(radii)