import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.model_bounds as model_bounds
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import double_integrator

import torch
import torch.nn as nn
import numpy as np
import copy
import gurobipy
import matplotlib.pyplot as plt


def get_value_function():
    dt = 1.
    dtype = torch.float64
    (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
    x_dim = A_c.shape[1]
    u_dim = B_c.shape[1]
    A = torch.eye(x_dim, dtype=dtype) + dt * A_c
    B = dt * B_c
    sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
    c = torch.zeros(x_dim, dtype=dtype)
    P = torch.cat((-torch.eye(x_dim+u_dim),
                   torch.eye(x_dim+u_dim)), 0).type(dtype)
    x_lo = -10. * torch.ones(x_dim, dtype=dtype)
    x_up = 10. * torch.ones(x_dim, dtype=dtype)
    u_lo = -1. * torch.ones(u_dim, dtype=dtype)
    u_up = 1. * torch.ones(u_dim, dtype=dtype)
    q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
    sys.add_mode(A, B, c, P, q)
    xN = torch.Tensor([0., 0.]).type(dtype)
    R = torch.eye(sys.u_dim)
    N = 5
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    vf.set_cost(R=R)
    vf.set_terminal_cost(Rt=R)
    vf.set_constraints(xN=xN)
    return vf


if __name__=="__main__":
    gen_samples = False
    x_samples_file = 'data/robust_value_demo_x'
    v_samples_file = 'data/robust_value_demo_v'
    vf = get_value_function()
    x0_lo = -1. * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
    x0_up = 1. * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
    if gen_samples:
        num_breaks = [5] * vf.sys.x_dim
        x_samples_train, v_samples_train = vf.get_sample_grid(x0_lo, x0_up, 
                                                              num_breaks)
        torch.save(x_samples_train, x_samples_file + '_train.pt')
        torch.save(v_samples_train, v_samples_file + '_train.pt')
    else:
        x_samples_train = torch.load(x_samples_file + '_train.pt')
        v_samples_train = torch.load(v_samples_file + '_train.pt')
    # train set and test sets
    batch_size=100
    train_data_set = torch.utils.data.TensorDataset(x_samples_train, 
                                                    v_samples_train)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, 
                                                    batch_size=batch_size, 
                                                    shuffle=True)
    nn_width = 16
    nn_depth = 1
    nn_layers = [nn.Linear(vf.sys.x_dim, nn_width), nn.ReLU()]
    for i in range(nn_depth):
        nn_layers += [nn.Linear(nn_width, nn_width), nn.ReLU()]
    nn_layers += [nn.Linear(nn_width, 1)]    
    num_epoch = 1000
    learning_rate = 1e-3
    model = nn.Sequential(*nn_layers).double()    
    l2_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate, momentum=.25)
    robust_model = copy.deepcopy(model)  
    robust_l2_fn = torch.nn.MSELoss(reduction="sum") 
    robust_optimizer = torch.optim.SGD(robust_model.parameters(),
                                       lr=learning_rate, momentum=.25)
    l2_log = []
    epsilon_log = []
    robust_l2_log = []
    robust_epsilon_log = []
    mb = model_bounds.ModelBounds(robust_model, vf)
    pre_training_iter = 500
    n_iter = 0
    for epoch in range(num_epoch):
        for batch_data, batch_label in train_data_loader:
            # nonrobust (pure l2 regression)
            y_pred = model(batch_data)
            l2 = l2_fn(y_pred, batch_label) / batch_size
            if n_iter > l2_training_iter:
                (Q1, Q2, q1, q2, k,
                 G1, G2, h,
                 A1, A2, b) = mb.upper_bound_opt(model, x0_lo, x0_up)
                mb_prob = gurobi_torch_mip.GurobiTorchMIQP(vf.dtype)
                mb_prob.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
                mb_prob.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions, 2)
                mb_prob.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
                y = mb_prob.addVars(Q1.shape[0], lb=-gurobipy.GRB.INFINITY, 
                                    vtype=gurobipy.GRB.CONTINUOUS, name="y")
                gamma = mb_prob.addVars(Q2.shape[0], vtype=gurobipy.GRB.BINARY, 
                                        name="gamma")
                mb_prob.setObjective([.5 * Q1, .5* Q2], 
                                     [(y, y), (gamma, gamma)],
                                     [q1, q2], [y, gamma], k,
                                     gurobipy.GRB.MINIMIZE)
                mb_prob.gurobi_model.remove(mb_prob.gurobi_model.getConstrs())
                for i in range(G1.shape[0]):
                    mb_prob.addLConstr([G1[i,:], G2[i,:]], [y, gamma],
                                       gurobipy.GRB.LESS_EQUAL, h[i])
                for i in range(A1.shape[0]):
                    mb_prob.addLConstr([A1[i,:], A2[i,:]], [y, gamma],
                                       gurobipy.GRB.EQUAL, b[i])
                mb_prob.gurobi_model.update()
                mb_prob.gurobi_model.optimize()
                epsilon = mb_prob.compute_objective_from_mip_data_and_solution(penalty=1e-8)
            else:
                epsilon = torch.Tensor([0.]).type(vf.dtype)
            loss = l2 + 0.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l2_log.append(l2.item())
            epsilon_log.append(torch.clamp(-epsilon, 0.).item())
            # robust regression
            y_pred_robust = robust_model(batch_data)
            robust_l2 = robust_l2_fn(y_pred_robust, batch_label) / batch_size
            if n_iter > l2_training_iter:
                (Q1, Q2, q1, q2, k,
                 G1, G2, h,
                 A1, A2, b) = mb.upper_bound_opt(robust_model, x0_lo, x0_up)
                mb_prob = gurobi_torch_mip.GurobiTorchMIQP(vf.dtype)
                mb_prob.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
                mb_prob.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions, 2)
                mb_prob.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
                y = mb_prob.addVars(Q1.shape[0], lb=-gurobipy.GRB.INFINITY, 
                                    vtype=gurobipy.GRB.CONTINUOUS, name="y")
                gamma = mb_prob.addVars(Q2.shape[0], vtype=gurobipy.GRB.BINARY, 
                                        name="gamma")
                mb_prob.setObjective([.5 * Q1, .5* Q2], 
                                     [(y, y), (gamma, gamma)],
                                     [q1, q2], [y, gamma], k,
                                     gurobipy.GRB.MINIMIZE)
                mb_prob.gurobi_model.remove(mb_prob.gurobi_model.getConstrs())
                for i in range(G1.shape[0]):
                    mb_prob.addLConstr([G1[i,:], G2[i,:]], [y, gamma],
                                       gurobipy.GRB.LESS_EQUAL, h[i])
                for i in range(A1.shape[0]):
                    mb_prob.addLConstr([A1[i,:], A2[i,:]], [y, gamma],
                                       gurobipy.GRB.EQUAL, b[i])
                mb_prob.gurobi_model.update()
                mb_prob.gurobi_model.optimize()
                robust_epsilon = mb_prob.compute_objective_from_mip_data_and_solution(penalty=1e-8)
            else:
                robust_epsilon = torch.Tensor([0.]).type(vf.dtype)
            robust_loss = robust_l2 + .1*torch.exp(-robust_epsilon)
            robust_optimizer.zero_grad()
            robust_loss.backward()
            robust_optimizer.step()
            robust_l2_log.append(robust_l2.item())
            robust_epsilon_log.append(torch.clamp(-robust_epsilon, 0.).item())
            print("+++")
            print(n_iter)
            print(l2.item())
            print(-epsilon.item())
            print(robust_l2.item())
            print(-robust_epsilon.item())
            print("===")
            n_iter += 1
    epsilon_log[:l2_training_iter+1] = [None] * (l2_training_iter+1)
    robust_epsilon_log[:l2_training_iter+1] = [None] * (l2_training_iter+1)
    plt.plot(l2_log)
    plt.plot(epsilon_log)
    plt.plot(robust_l2_log)
    plt.plot(robust_epsilon_log)
    plt.legend(['l2', 'epsilon', 'robust l2', 'robust epsilon'])
    plt.show()