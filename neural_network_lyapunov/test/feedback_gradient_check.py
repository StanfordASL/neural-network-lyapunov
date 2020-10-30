import torch
import gurobipy
import numpy as np
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.train_lyapunov as train_lyapunov


def check_sample_loss_grad(
        lyap, V_lambda, x_equilibrium, x_samples, atol, rtol):
    utils.network_zero_grad(lyap.lyapunov_relu)
    utils.network_zero_grad(lyap.system.controller_network)
    dut = train_lyapunov.TrainLyapunovReLU(
        lyap, V_lambda, x_equilibrium)
    x_next_samples = torch.cat([lyap.system.step_forward(
        x_samples[i]).reshape((1, -1)) for i in range(x_samples.shape[0])],
        dim=0)
    total_loss = dut.total_loss(
        x_samples, x_samples, x_next_samples, 1., 1., None, None)[0]
    total_loss.backward()
    controller_grad = utils.extract_relu_parameters_grad(
        lyap.system.controller_network)
    lyapunov_grad = utils.extract_relu_parameters_grad(
        lyap.lyapunov_relu)

    def compute_loss(
            controller_params: np.ndarray, lyapunov_params: np.ndarray):
        with torch.no_grad():
            utils.update_relu_params(
                lyap.system.controller_network,
                torch.from_numpy(controller_params))
            utils.update_relu_params(
                lyap.lyapunov_relu,
                torch.from_numpy(lyapunov_params))
            x_next_samples = torch.cat([
                lyap.system.step_forward(
                    x_samples[i]).reshape((1, -1)) for i in
                range(x_samples.shape[0])], dim=0)
            return dut.total_loss(
                x_samples, x_samples, x_next_samples, 1., 1., None, None
                )[0].item()

    grad_numerical = utils.compute_numerical_gradient(
        compute_loss, utils.extract_relu_parameters(
            lyap.system.controller_network).detach().numpy(),
        utils.extract_relu_parameters(
            lyap.lyapunov_relu).detach().numpy())
    np.testing.assert_allclose(
        controller_grad, grad_numerical[0], rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        lyapunov_grad, grad_numerical[1], rtol=rtol, atol=atol)


def check_lyapunov_mip_loss_grad(
    lyap, x_equilibrium, V_lambda, V_epsilon,
        positivity_flag, atol, rtol):
    utils.network_zero_grad(lyap.lyapunov_relu)
    utils.network_zero_grad(lyap.system.controller_network)
    dut = train_lyapunov.TrainLyapunovReLU(
        lyap, V_lambda, x_equilibrium)
    x_dim = x_equilibrium.shape[0]
    x_samples = torch.empty((0, x_dim), dtype=torch.float64)
    x_next_samples = torch.empty((0, x_dim), dtype=torch.float64)
    if positivity_flag:
        dut.lyapunov_positivity_mip_pool_solutions = 1
        dut.lyapunov_positivity_epsilon = V_epsilon
        total_loss = dut.total_loss(
            x_samples, x_samples, x_next_samples, 0., 0., 1., None)[0]
    else:
        dut.lyapunov_derivative_mip_pool_solutions = 1
        dut.lyapunov_derivative_epsilon = V_epsilon
        total_loss = dut.total_loss(
            x_samples, x_samples, x_next_samples, 0., 0., None, 1.)[0]
    total_loss.backward()

    controller_grad = utils.extract_relu_parameters_grad(
        lyap.system.controller_network)
    lyapunov_grad = utils.extract_relu_parameters_grad(lyap.lyapunov_relu)

    def compute_loss(
            controller_params: np.ndarray, lyapunov_params: np.ndarray):
        with torch.no_grad():
            utils.update_relu_params(
                dut.lyapunov_hybrid_system.system.controller_network,
                torch.from_numpy(controller_params))
            utils.update_relu_params(
                dut.lyapunov_hybrid_system.lyapunov_relu,
                torch.from_numpy(lyapunov_params))
            x_next_samples = torch.empty((0, x_dim), dtype=torch.float64)
            if positivity_flag:
                dut.lyapunov_positivity_epsilon = V_epsilon
                return dut.total_loss(
                    x_samples, x_samples, x_next_samples, 0., 0., 1.,
                    None)[0].item()
            else:
                dut.lyapunov_derivative_epsilon = V_epsilon
                return dut.total_loss(
                    x_samples, x_samples, x_next_samples, 0., 0., None,
                    1.)[0].item()

    grad_numerical = utils.compute_numerical_gradient(
        compute_loss, utils.extract_relu_parameters(
            lyap.system.controller_network).detach().numpy(),
        utils.extract_relu_parameters(
            lyap.lyapunov_relu).detach().numpy())
    np.testing.assert_allclose(
        controller_grad.detach().numpy(), grad_numerical[0], rtol=rtol,
        atol=atol)
    np.testing.assert_allclose(
        lyapunov_grad.detach().numpy(), grad_numerical[1], rtol=rtol,
        atol=atol)


def create_mip(
    lyap, x_equilibrium, V_lambda, V_epsilon, positivity_flag, eps_type,
        controller_param, lyapunov_param):
    utils.update_relu_params(
        lyap.system.controller_network, _to_tensor(controller_param))
    utils.network_zero_grad(lyap.system.controller_network)
    utils.update_relu_params(lyap.lyapunov_relu, _to_tensor(lyapunov_param))
    utils.network_zero_grad(lyap.lyapunov_relu)
    if positivity_flag:
        mip = lyap.lyapunov_positivity_as_milp(
            x_equilibrium, V_lambda, V_epsilon)[0]
    else:
        mip = lyap.lyapunov_derivative_as_milp(
            x_equilibrium, V_lambda, V_epsilon, eps_type)[0]
    return mip


def check_lyapunov_mip_grad(
    lyap, x_equilibrium, V_lambda, V_epsilon, positivity_flag, eps_type, atol,
        rtol):
    """
    For each term in the MIP (constraint matrix, cost vector, constraint rhs,
    etc), check if the gradient is correct.
    """

    # Check the gradient of one entry in the MIP
    def check_mip_entry_grad(entry_name, entry_index):
        print(f"{entry_name}[{entry_index}]")
        def get_entry(
                controller_param, lyapunov_param, entry_name, entry_index):
            if isinstance(controller_param, torch.Tensor):
                controller_param_torch = controller_param
            elif isinstance(controller_param, np.ndarray):
                controller_param_torch = torch.from_numpy(controller_param)
            if isinstance(lyapunov_param, torch.Tensor):
                lyapunov_param_torch = lyapunov_param
            elif isinstance(lyapunov_param, np.ndarray):
                lyapunov_param_torch = torch.from_numpy(lyapunov_param)
            mip_tmp = create_mip(
                lyap, x_equilibrium, V_lambda, V_epsilon, positivity_flag,
                eps_type, controller_param_torch, lyapunov_param_torch)
            if entry_index is None:
                return getattr(mip_tmp, entry_name)
            elif entry_index == "sum":
                entries = getattr(mip_tmp, entry_name)
                if isinstance(entries, list):
                    return torch.stack(entries).sum()
                elif isinstance(entries, torch.Tensor):
                    return entries.sum()
            else:
                return getattr(mip_tmp, entry_name)[entry_index]

        check_lyapunov_grad(lyap, lambda p1, p2: get_entry(
            p1, p2, entry_name, entry_index), atol, rtol, dx=1e-7)

    controller_relu_params = utils.extract_relu_parameters(
        lyap.system.controller_network).clone()
    lyapunov_relu_params = utils.extract_relu_parameters(
        lyap.lyapunov_relu).clone()
    mip = create_mip(
        lyap, x_equilibrium, V_lambda, V_epsilon, positivity_flag, eps_type,
        controller_relu_params, lyapunov_relu_params)
    check_mip_entry_grad("Ain_r_val", "sum")
    check_mip_entry_grad("Ain_zeta_val", "sum")
    check_mip_entry_grad("rhs_in", "sum")
    check_mip_entry_grad("Aeq_r_val", "sum")
    check_mip_entry_grad("Aeq_zeta_val", "sum")
    check_mip_entry_grad("rhs_eq", "sum")
    check_mip_entry_grad("c_r", "sum")
    check_mip_entry_grad("c_zeta", "sum")
    check_mip_entry_grad("c_constant", None)
    for i in range(len(mip.Ain_r_val)):
        check_mip_entry_grad("Ain_r_val", i)
    for i in range(len(mip.Ain_zeta_val)):
        check_mip_entry_grad("Ain_zeta_val", i)
    for i in range(len(mip.rhs_in)):
        check_mip_entry_grad("rhs_in", i)
    for i in range(len(mip.Aeq_r_val)):
        check_mip_entry_grad("Aeq_r_val", i)
    for i in range(len(mip.Aeq_zeta_val)):
        check_mip_entry_grad("Aeq_zeta_val", i)
    for i in range(len(mip.rhs_eq)):
        check_mip_entry_grad("rhs_eq", i)
    for i in range(len(mip.c_r)):
        check_mip_entry_grad("c_r", i)
    for i in range(len(mip.c_zeta)):
        check_mip_entry_grad("c_zeta", i)


def check_lyapunov_grad(lyap, eval_fun, atol, rtol, dx):
    """
    @param eval_fun takes in the controller relu parameter and lyapunov relu
    parameter, returns a torch 0-dimensional tensor.
    """
    utils.network_zero_grad(lyap.system.controller_network)
    utils.network_zero_grad(lyap.lyapunov_relu)
    controller_relu_param = utils.extract_relu_parameters(
        lyap.system.controller_network)
    lyapunov_relu_param = utils.extract_relu_parameters(lyap.lyapunov_relu)

    val = eval_fun(controller_relu_param, lyapunov_relu_param)
    if val.requires_grad:
        val.backward()
        controller_grad = utils.extract_relu_parameters_grad(
            lyap.system.controller_network)
        lyapunov_grad = utils.extract_relu_parameters_grad(lyap.lyapunov_relu)
    else:
        print("doesn't require gradient")
        controller_grad = torch.zeros_like(controller_relu_param)
        lyapunov_grad = torch.zeros_like(lyapunov_relu_param)

    with torch.no_grad():
        grad_numerical = utils.compute_numerical_gradient(
            lambda p1, p2: eval_fun(p1, p2).item(),
            controller_relu_param.detach().numpy(),
            lyapunov_relu_param.detach().numpy(), dx=dx)

    np.testing.assert_allclose(
        controller_grad.detach().numpy(), grad_numerical[0], atol=atol,
        rtol=rtol)
    np.testing.assert_allclose(
        lyapunov_grad.detach().numpy(), grad_numerical[1], atol=atol,
        rtol=rtol)


def compute_mip_loss(
    lyap, x_equilibrium, V_lambda, V_epsilon, positivity_flag, eps_type,
        controller_relu_param, lyapunov_relu_param):
    mip = create_mip(
        lyap, x_equilibrium, V_lambda, V_epsilon, positivity_flag, eps_type,
        controller_relu_param, lyapunov_relu_param)
    mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
    mip.gurobi_model.optimize()

    mip.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber, 0)

    r_sol = torch.tensor([var.xn for var in mip.r], dtype=mip.dtype)
    zeta_sol = torch.tensor(
        [round(var.xn) for var in mip.zeta], dtype=mip.dtype)
    (Ain_r, Ain_zeta, rhs_in) = mip.get_inequality_constraints()
    lhs_in = Ain_r @ r_sol + Ain_zeta @ zeta_sol
    active_ineq_row_indices = np.arange(len(mip.rhs_in))
    active_ineq_row_indices = set(active_ineq_row_indices[np.abs(
        lhs_in.detach().numpy() - rhs_in.detach().numpy()) < 1E-6])
    A_act1 = torch.sparse.DoubleTensor(
        torch.LongTensor([mip.Aeq_r_row, mip.Aeq_r_col]),
        torch.stack(mip.Aeq_r_val).type(torch.float64),
        torch.Size([len(mip.rhs_eq), len(mip.r)])).type(mip.dtype).to_dense()
    Aeq_zeta = torch.sparse.DoubleTensor(
        torch.LongTensor([mip.Aeq_zeta_row, mip.Aeq_zeta_col]),
        torch.stack(mip.Aeq_zeta_val).type(torch.float64),
        torch.Size([len(mip.rhs_eq), len(mip.zeta)])).type(
            mip.dtype).to_dense()
    b_act1 = torch.stack([
        s.squeeze() for s in mip.rhs_eq]) - Aeq_zeta @ zeta_sol
    active_ineq_row_indices_list = list(active_ineq_row_indices)
    Ain_active_r = Ain_r[active_ineq_row_indices_list]
    Ain_active_zeta = Ain_zeta[active_ineq_row_indices_list]
    rhs_in_active = rhs_in[active_ineq_row_indices_list]
    b_act2 = rhs_in_active - Ain_active_zeta @ zeta_sol
    return A_act1.sum() + b_act1.sum() + Ain_active_r.sum() + b_act2.sum()


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
