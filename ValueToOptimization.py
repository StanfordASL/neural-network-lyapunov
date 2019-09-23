import torch

class ValueFunction():

    def __init__(self):
        pass

    def value_program(self, sys, 
                      Q, q, R, r, Z, z, 
                      Qt, qt, Rt, rt, Zt, zt, 
                      x_lo, x_up, u_lo, u_up, x0, xN, N):
        """
        Generates a trajectory optimization problem with the following parameters
        
        min ∑(.5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n] + rᵀu[n] + zᵀα[n])
                + .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N] + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤ rhs_in_dyn
        Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] ≤ rhs_eq_dyn
        x_lo ≤ x[n] ≤ x_up
        u_lo ≤ u[n] ≤ u_up
        x[0] == x0
        x[N] == xN
        α[N] ∈ Z
                
        the problem is returned in our standard MIQP form so that it can easily be passed to verification functions.
        Letting x = x[0], and s = x[1]...x[N]
        
        min .5 sᵀ Q2 s + .5 αᵀ Q3 α + q2ᵀ s + q3ᵀ α
        s.t. Ain1 x + Ain2 s + Ain3 α ≤ brhs_in
             Aeq1 x + Aeq2 s + Aeq3 α ≤ brhs_eq
             α ∈ Z
        """

        Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn, rhs_in_dyn = sys.get_dyn_in()
        Aeq1_dyn, Aeq2_dyn, Aeq3_dyn, Aeq4_dyn, rhs_eq_dyn = sys.get_dyn_eq()
        
        xdim = Q.shape[0]
        udim = R.shape[0]
        sdim = (xdim+udim)*(N-1)
        adim = Z.shape[0]
        alphadim = adim*(N-1)
        
        Q2 = torch.zeros(sdim,sdim)
        q2 = torch.zeros(sdim)
        Q3 = torch.zeros(alphadim,alphadim)
        q3 = torch.zeros(alphadim)
        for i in range(N-2):
            Q2[i*(xdim+udim):i*(xdim+udim)+xdim,i*(xdim+udim):i*(xdim+udim)+xdim] = Q
            Q2[i*(xdim+udim)+xdim:i*(xdim+udim)+xdim+udim,i*(xdim+udim)+xdim:i*(xdim+udim)+xdim+udim] = R
            q2[i*(xdim+udim):i*(xdim+udim)+xdim] = q
            q2[i*(xdim+udim)+xdim:i*(xdim+udim)+xdim+udim] = r
            Q3[i*adim:(i+1)*adim,i*adim:(i+1)*adim] = Z
            q3[i*adim:(i+1)*adim] = z
        i = N-2
        Q2[i*(xdim+udim):i*(xdim+udim)+xdim,i*(xdim+udim):i*(xdim+udim)+xdim] = Qt
        Q2[i*(xdim+udim)+xdim:i*(xdim+udim)+xdim+udim,i*(xdim+udim)+xdim:i*(xdim+udim)+xdim+udim] = Rt
        q2[i*(xdim+udim):i*(xdim+udim)+xdim] = qt
        q2[i*(xdim+udim)+xdim:i*(xdim+udim)+xdim+udim] = rt
        Q3[i*adim:(i+1)*adim,i*adim:(i+1)*adim] = Zt
        q3[i*adim:(i+1)*adim] = zt            
        
        Ain1 = torch.zeros(0,xdim)
        Ain2 = torch.zeros(0,sdim)
        Ain3 = torch.zeros(0,alphadim)
        brhs_in = torch.zeros(0)
        s_ineq = torch.cat((Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn), 1)
        for i in range(N-2):
            Ain2[,i*(xdim+udim):i*(xdim+udim)+2*(xdim+udim)] = s_ineq
            Ain3[,i*adim:(i+1)*adim] = Ain5_dyn
            
        return(Ain1, Ain2, Ain3, brhs_in, Aeq1, Aeq2, Aeq3, brhs_eq, Q2, Q3, q2, q3)
