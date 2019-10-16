import numpy as np
import cvxpy as cp

np.random.seed(10)

def main():
    ny = 3
    nx = 3
    y = cp.Variable(ny)
    x = cp.Variable(nx)
    
    c1 = np.random.rand(ny)
    c2 = np.random.rand(nx)
    A1 = np.random.rand(ny,ny)
    A2 = np.random.rand(nx,nx)
    A3 = np.random.rand(ny,ny)
    A3 = np.random.rand(nx,nx)
    
    y0 = np.random.rand(ny)
    x0 = np.random.rand(nx)
    
    b1 = A1@y + 1e-2
    b2 = A2@x + 1e-2
    b3 = A3@x + A4@y + 1e-2
    
    maxiter = 10
    for iter in range(maxiter):
         
    
if __name__=='__main__':
    main()