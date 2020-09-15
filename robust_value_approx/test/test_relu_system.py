import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import robust_value_approx.relu_system as relu_system
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip


def setup_relu_dyn(dtype, params=None):
    # Construct a simple ReLU model with 2 hidden layers
    # params is the value of weights/bias after concatenation.
    # the network has the same number of outputs as inputs (2)
    if params is not None:
        assert(isinstance(params, torch.Tensor))
        assert(params.shape == (35,))
    linear1 = nn.Linear(2, 3)
    if params is None:
        linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                           dtype=dtype)
        linear1.bias.data = torch.tensor([-11, 10, 5], dtype=dtype)
    else:
        linear1.weight.data = params[:6].clone().reshape((3, 2))
        linear1.bias.data = params[6:9].clone()
    linear2 = nn.Linear(3, 4)
    if params is None:
        linear2.weight.data = torch.tensor(
                [[-1, -0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1.5, 4, 6]],
                dtype=dtype)
        linear2.bias.data = torch.tensor([-3, 2, 0.7, 1.5], dtype=dtype)
    else:
        linear2.weight.data = params[9:21].clone().reshape((4, 3))
        linear2.bias.data = params[21:25].clone()
    linear3 = nn.Linear(4, 2)
    if params is None:
        linear3.weight.data = torch.tensor([[4, 5, 6, 7], [8, 7, 5.5, 4.5]],
                                           dtype=dtype)
        linear3.bias.data = torch.tensor([-9, 3], dtype=dtype)
    else:
        linear3.weight.data = params[25:33].clone().reshape((2, 4))
        linear3.bias.data = params[33:35].clone().reshape((2))
    relu1 = nn.Sequential(
        linear1, nn.ReLU(), linear2, nn.ReLU(), linear3)
    assert(not relu1.forward(torch.tensor([0, 0], dtype=dtype))[0].item() == 0)
    assert(not relu1.forward(torch.tensor([0, 0], dtype=dtype))[1].item() == 0)
    return relu1


class TestAutonomousReluSystem(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.float64
        self.relu_dyn = setup_relu_dyn(self.dtype)
        self.x_lo = torch.tensor([-1e4, -1e4], dtype=self.dtype)
        self.x_up = torch.tensor([1e4, 1e4], dtype=self.dtype)
        self.system = relu_system.AutonomousReLUSystem(self.dtype,
                                                       self.x_lo, self.x_up,
                                                       self.relu_dyn)

    def test_relu_system_as_milp(self):
        (Aout_s, Cout,
         Ain_x, Ain_s, Ain_gamma, rhs_in,
         Aeq_x, Aeq_s, Aeq_gamma, rhs_eq) = \
            self.system.mixed_integer_constraints()

        def check_transition(x0):
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(
                self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="x")
            s = milp.addVars(
                Ain_s.shape[1], lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="s")
            gamma = milp.addVars(
                Ain_gamma.shape[1], lb=0., vtype=gurobipy.GRB.BINARY,
                name="gamma")
            if rhs_in.shape[0] > 0:
                milp.addMConstrs(
                    [Ain_x, Ain_s, Ain_gamma], [x, s, gamma],
                    sense=gurobipy.GRB.LESS_EQUAL, b=rhs_in.squeeze(),
                    name="relu_dynamics_ineq")
            if rhs_eq.shape[0] > 0:
                milp.addMConstrs(
                    [Aeq_x, Aeq_s, Aeq_gamma], [x, s, gamma],
                    sense=gurobipy.GRB.EQUAL, b=rhs_eq.squeeze(),
                    name="relu_dynamics_eq")
            for i in range(self.system.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x0[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.optimize()

            s_val = torch.tensor([si.X for si in s], dtype=self.dtype)
            x_next_val = Aout_s @ s_val + Cout
            np.testing.assert_array_almost_equal(x_next_val.detach().numpy(),
                                                 self.relu_dyn(
                                                    x0).detach().numpy(),
                                                 decimal=5)

        torch.manual_seed(0)

        for i in range(10):
            x0 = torch.rand(2, dtype=self.dtype)
            check_transition(x0)


class TestReLUSystem(unittest.TestCase):
    def test_mixed_integer_constraints(self):
        # Construct a ReLU system with nx = 2 and nu = 1
        self.dtype = torch.float64
        linear1 = torch.nn.Linear(3, 5, bias=True)
        linear1.weight.data = torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, -0.2, 0.4], [0.1, 0.3, -1.2],
             [1.5, 0.3, 0.3], [0.2, 1.5, 0.1]], dtype=self.dtype)
        linear1.bias.data = torch.tensor(
            [0.1, -1.2, 0.3, 0.2, -0.5], dtype=self.dtype)
        linear2 = torch.nn.Linear(5, 2, bias=True)
        linear2.weight.data = torch.tensor(
            [[0.1, -2.3, 1.5, 0.4, 0.2], [0.1, -1.2, -1.3, 0.3, 0.8]],
            dtype=self.dtype)
        linear2.bias.data = torch.tensor([0.2, -1.4], dtype=self.dtype)
        dynamics_relu = torch.nn.Sequential(
            linear1, torch.nn.LeakyReLU(0.1), linear2)

        x_lo = torch.tensor([-2, -2], dtype=self.dtype)
        x_up = torch.tensor([2, 2], dtype=self.dtype)
        u_lo = torch.tensor([-1], dtype=self.dtype)
        u_up = torch.tensor([1], dtype=self.dtype)
        dut = relu_system.ReLUSystem(
            self.dtype, x_lo, x_up, u_lo, u_up, dynamics_relu)
        self.assertEqual(dut.x_dim, 2)
        self.assertEqual(dut.u_dim, 1)

        (Aout_s, Cout,
         Ain_x, Ain_u, Ain_s, Ain_gamma, rhs_in,
         Aeq_x, Aeq_u, Aeq_s, Aeq_gamma, rhs_eq) = \
            dut.mixed_integer_constraints()

        def check_transition(x_val, u_val):
            milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
            x = milp.addVars(
                dut.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="x")
            u = milp.addVars(
                dut.u_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="u")
            s = milp.addVars(
                Ain_s.shape[1], lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="s")
            gamma = milp.addVars(
                Ain_gamma.shape[1], lb=0., vtype=gurobipy.GRB.BINARY,
                name="gamma")
            if rhs_in.shape[0] > 0:
                milp.addMConstrs(
                    [Ain_x, Ain_u, Ain_s, Ain_gamma], [x, u, s, gamma],
                    sense=gurobipy.GRB.LESS_EQUAL, b=rhs_in.squeeze(),
                    name="relu_dynamics_ineq")
            if rhs_eq.shape[0] > 0:
                milp.addMConstrs(
                    [Aeq_x, Aeq_u, Aeq_s, Aeq_gamma], [x, u, s, gamma],
                    sense=gurobipy.GRB.EQUAL, b=rhs_eq.squeeze(),
                    name="relu_dynamics_eq")
            for i in range(dut.x_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[x[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=x_val[i])
            for i in range(dut.u_dim):
                milp.addLConstr(
                    [torch.tensor([1.], dtype=self.dtype)], [[u[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=u_val[i])
            milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
            milp.gurobi_model.setParam(gurobipy.GRB.Param.DualReductions, 0)
            milp.gurobi_model.optimize()

            s_val = torch.tensor([si.X for si in s], dtype=self.dtype)
            x_next_val = Aout_s @ s_val + Cout
            x_next_val_expected = dynamics_relu(torch.cat((x_val, u_val)))
            np.testing.assert_array_almost_equal(
                x_next_val.detach().numpy(),
                x_next_val_expected.detach().numpy(), decimal=5)

        check_transition(
            torch.tensor([0.2, 0.5], dtype=self.dtype),
            torch.tensor([0.1], dtype=self.dtype))
        check_transition(
            torch.tensor([1.2, 0.5], dtype=self.dtype),
            torch.tensor([0.1], dtype=self.dtype))
        check_transition(
            torch.tensor([-1.2, 0.3], dtype=self.dtype),
            torch.tensor([0.5], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
