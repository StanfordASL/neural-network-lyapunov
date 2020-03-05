import torch
import numpy as np
import plotly.graph_objs as go


class NonlinearSystem:
    def dyn(self, var, arraylib=np):
	    x_dim = self.x_dim[0]
	    u_dim = self.u_dim[0]
	    x0 = var[:x_dim]
	    u0 = var[x_dim:x_dim+u_dim]
	    dt0 = var[x_dim+u_dim:x_dim+u_dim+1]
	    x1 = var[x_dim+u_dim+1:x_dim+u_dim+1+x_dim]
	    u1 = var[x_dim+u_dim+1+x_dim:x_dim+u_dim+1+x_dim+u_dim]
	    dx0 = self.dx(x0, u0, arraylib=arraylib)
	    dx1 = self.dx(x1, u1, arraylib=arraylib)
	    return x0 + .5 * (dx0 + dx1) * dt0 - x1

    def quad_cost(self, var,
                  Q=None, R=None, x_desired=None, u_desired=None,
                  arraylib=np):
        x_dim = self.x_dim[0]
        u_dim = self.u_dim[0]
        x = var[:x_dim]
        u = var[x_dim:x_dim+u_dim]
        dt = var[x_dim+u_dim:x_dim+u_dim+1]
        if Q is None:
            Q = arraylib.zeros((x_dim, x_dim))
        if R is None:
            R = arraylib.zeros((u_dim, u_dim))
        if x_desired is None:
            x_desired = arraylib.zeros(x_dim)
        if u_desired is None:
            u_desired = arraylib.zeros(u_dim)
        cost = dt[0]*((x - x_desired)@Q@(x - x_desired) + (u - u_desired)@R@(u - u_desired))
        return cost

    def plot_result_named(self, result, names):
        fig = go.Figure()
        x = torch.cat([x.unsqueeze(0) for x in result['x_traj']]).t()
        t = torch.cat([torch.zeros(1, dtype=x.dtype), torch.cumsum(
                result['dt_traj'], 0).squeeze()[:-1]])
        for i in range(self.x_dim[0]):
            fig.add_trace(go.Scatter(
                x=t,
                y=x[i, :],
                name=names[i],
                ))
        return fig