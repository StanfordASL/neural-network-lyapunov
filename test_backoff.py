import numpy as np
import cvxpy as cp

np.random.seed(10)

def main():
    n = 5
    ncon = 3
    
    A = np.random.rand(ncon,n)
    x_sol = np.random.rand(n)
    b = A@x_sol + 1e-2
    
    M = np.random.rand(n,n)
    M = M.T@M + np.eye(n)*.1
    m = np.random.rand(n)
    
    x = cp.Variable(n)
    
    obj = cp.quad_form(x,M) - 2.*m.T@x
    con = [A@x <= b]
    prob = cp.Problem(cp.Minimize(obj),con)
    prob.solve(solver=cp.CPLEX,verbose=False)
    c = obj.value
    print(c)
    print([c.value() for c in con])
    
    backoff = .1
    v = c + backoff*np.abs(c)
    backoff_obj = cp.Minimize(0.)
    t = cp.Variable(1)
    L = np.linalg.cholesky(M)
    cone_con = [cp.SOC(t, L@(x - np.linalg.inv(M)@m)), t == np.sqrt(v + m.T@np.linalg.inv(M)@m)]
    backoff_con = con + cone_con
    prob = cp.Problem(backoff_obj, backoff_con)
    prob.solve(solver=cp.CPLEX,verbose=False)
    print(obj.value)
    print([c.value() for c in con])

if __name__=='__main__':
    main()