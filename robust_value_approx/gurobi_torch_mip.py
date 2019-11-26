import gurobipy
import torch
import numpy as np


class GurobiTorchMIP:
    """
    This class will be used in computing the gradient of an MIP optimal cost
    w.r.t constraint/objective data. It uses gurobi to solve the MIP, but also
    stores the constraint/objective data in pytorch tensor format, so that we
    can run automatic differentiation.

    Internally it stores an MIP
    min cost
    s.t Ain_r * r + Ain_zeta * ζ <= rhs_in
        Aeq_r * r + Aeq_zeta * ζ = rhs_eq
    where r includes all continuous variables, and ζ includes all binary
    variables.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.gurobi_model = gurobipy.Model()
        self.r = []
        self.zeta = []
        self.Ain_r_row = []
        self.Ain_r_col = []
        self.Ain_r_val = []
        self.Ain_zeta_row = []
        self.Ain_zeta_col = []
        self.Ain_zeta_val = []
        self.rhs_in = []
        self.Aeq_r_row = []
        self.Aeq_r_col = []
        self.Aeq_r_val = []
        self.Aeq_zeta_row = []
        self.Aeq_zeta_col = []
        self.Aeq_zeta_val = []
        self.rhs_eq = []
        # r_indices[var] maps a gurobi continuous variable to its index in r.
        # Namely self.r[r_indices[var]] = var
        self.r_indices = {}
        # zeta_indices[var] maps a gurobi binary variable to its index in zeta.
        # Namely self.zeta[zeta_indices[var]] = var
        self.zeta_indices = {}

    def addVars(self, num_vars, lb=0, ub=gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="x"):
        """
        @return new_vars_list A list of new variables.
        """
        new_vars = self.gurobi_model.addVars(
            num_vars, lb=lb, vtype=vtype, name=name)
        self.gurobi_model.update()
        if vtype == gurobipy.GRB.CONTINUOUS:
            num_existing_r = len(self.r_indices)
            self.r.extend([new_vars[i] for i in range(num_vars)])
            for i in range(num_vars):
                self.r_indices[new_vars[i]] = num_existing_r + i
            # If lower bound is not -inf, then add the inequality constraint
            # x>lb
            if lb != -gurobipy.GRB.INFINITY:
                for i in range(num_vars):
                    self.Ain_r_row.append(len(self.rhs_in))
                    self.Ain_r_col.append(num_existing_r + i)
                    self.Ain_r_val.append(torch.tensor(-1, dtype=self.dtype))
                    self.rhs_in.append(torch.tensor(-lb, dtype=self.dtype))
            if ub != gurobipy.GRB.INFINITY:
                for i in range(num_vars):
                    self.Ain_r_row.append(len(self.rhs_in))
                    self.Ain_r_col.append(num_existing_r + i)
                    self.Ain_r_val.append(torch.tensor(1, dtype=self.dtype))
                    self.rhs_in.append(torch.tensor(ub, dtype=self.dtype))
        elif vtype == gurobipy.GRB.BINARY:
            num_existing_zeta = len(self.zeta_indices)
            self.zeta.extend([new_vars[i] for i in range(num_vars)])
            for i in range(num_vars):
                self.zeta_indices[new_vars[i]] = num_existing_zeta + i
        else:
            raise Exception("Only support continuous or binary variables")
        return [new_vars[i] for i in range(num_vars)]

    def addLConstr(self, coeffs, variables, sense, rhs, name=""):
        """
        Add linear constraint.
        @param coeffs A list of 1D pytorch tensors. coeffs[i] are the
        coefficients for variables[i]
        @param variables A list of lists. variables[i] is a list of gurobi
        variables. Note that the variables cannot overlap.
        @param sense GRB.EQUAL, GRB.LESS_EQUAL or GRB.GREATER_EQUAL
        @param rhs The right-hand side of the constraint.
        @param name The name of the constraint.
        @return new constraint object.
        """
        if isinstance(rhs, torch.Tensor):
            rhs_tensor = rhs
        else:
            assert(isinstance(rhs, float))
            rhs_tensor = torch.tensor(rhs, dtype=self.dtype)
        expr = 0
        assert(isinstance(coeffs, list))
        assert(len(coeffs) == len(variables))
        num_vars = 0
        for coeff, var in zip(coeffs, variables):
            assert(isinstance(coeff, torch.Tensor))
            expr += gurobipy.LinExpr(coeff.tolist(), var)
            num_vars += len(var)
        constr = self.gurobi_model.addLConstr(
            expr, sense=sense, rhs=rhs_tensor, name=name)
        # r_used_flag[i] records if r[i] has appeared in @p variables.
        r_used_flag = [False] * len(self.r)
        zeta_used_flag = [False] * len(self.zeta)
        # First allocate memory
        if sense == gurobipy.GRB.EQUAL:
            new_Aeq_r_row = [None] * num_vars
            new_Aeq_r_col = [None] * num_vars
            new_Aeq_r_val = [None] * num_vars
            new_Aeq_zeta_row = [None] * num_vars
            new_Aeq_zeta_col = [None] * num_vars
            new_Aeq_zeta_val = [None] * num_vars
        else:
            new_Ain_r_row = [None] * num_vars
            new_Ain_r_col = [None] * num_vars
            new_Ain_r_val = [None] * num_vars
            new_Ain_zeta_row = [None] * num_vars
            new_Ain_zeta_col = [None] * num_vars
            new_Ain_zeta_val = [None] * num_vars

        # num_cont_vars is the number of continuous variables in this linear
        # constraint.
        num_cont_vars = 0
        # num_bin_vars is the number of binary variables in this linear
        # constraint.
        num_bin_vars = 0
        for coeff, var in zip(coeffs, variables):
            for i in range(len(var)):
                if var[i] in self.r_indices.keys():
                    r_index = self.r_indices[var[i]]
                    if r_used_flag[r_index]:
                        raise Exception("addLConstr: variable " + var[i].name
                                        + " is duplicated.")
                    r_used_flag[r_index] = True
                    if sense == gurobipy.GRB.EQUAL:
                        new_Aeq_r_row[num_cont_vars] = len(self.rhs_eq)
                        new_Aeq_r_col[num_cont_vars] = r_index
                        new_Aeq_r_val[num_cont_vars] = coeff[i]
                    else:
                        new_Ain_r_row[num_cont_vars] = len(self.rhs_in)
                        new_Ain_r_col[num_cont_vars] = r_index
                        new_Ain_r_val[num_cont_vars] = coeff[i] if\
                            sense == gurobipy.GRB.LESS_EQUAL else -coeff[i]
                    num_cont_vars += 1
                elif var[i] in self.zeta_indices.keys():
                    zeta_index = self.zeta_indices[var[i]]
                    if zeta_used_flag[zeta_index]:
                        raise Exception("addLConstr: variable " + var[i].name
                                        + " is duplicated.")
                    zeta_used_flag[zeta_index] = True
                    if sense == gurobipy.GRB.EQUAL:
                        new_Aeq_zeta_row[num_bin_vars] = len(self.rhs_eq)
                        new_Aeq_zeta_col[num_bin_vars] = zeta_index
                        new_Aeq_zeta_val[num_bin_vars] = coeff[i]
                    else:
                        new_Ain_zeta_row[num_bin_vars] = len(self.rhs_in)
                        new_Ain_zeta_col[num_bin_vars] = zeta_index
                        new_Ain_zeta_val[num_bin_vars] = coeff[i] if\
                            sense == gurobipy.GRB.LESS_EQUAL else -coeff[i]
                    num_bin_vars += 1
                else:
                    raise Exception("addLConstr: unknown variable " +
                                    var[i].name)
        if sense == gurobipy.GRB.EQUAL:
            if num_cont_vars > 0:
                self.Aeq_r_row.extend(new_Aeq_r_row[:num_cont_vars])
                self.Aeq_r_col.extend(new_Aeq_r_col[:num_cont_vars])
                self.Aeq_r_val.extend(new_Aeq_r_val[:num_cont_vars])
            if num_bin_vars > 0:
                self.Aeq_zeta_row.extend(new_Aeq_zeta_row[:num_bin_vars])
                self.Aeq_zeta_col.extend(new_Aeq_zeta_col[:num_bin_vars])
                self.Aeq_zeta_val.extend(new_Aeq_zeta_val[:num_bin_vars])
            self.rhs_eq.append(rhs_tensor)
        else:
            if num_cont_vars > 0:
                self.Ain_r_row.extend(new_Ain_r_row[:num_cont_vars])
                self.Ain_r_col.extend(new_Ain_r_col[:num_cont_vars])
                self.Ain_r_val.extend(new_Ain_r_val[:num_cont_vars])
            if num_bin_vars > 0:
                self.Ain_zeta_row.extend(new_Ain_zeta_row[:num_bin_vars])
                self.Ain_zeta_col.extend(new_Ain_zeta_col[:num_bin_vars])
                self.Ain_zeta_val.extend(new_Ain_zeta_val[:num_bin_vars])
            self.rhs_in.append(rhs_tensor if sense == gurobipy.GRB.LESS_EQUAL
                               else -rhs_tensor)

        return constr

    def get_active_constraints(self, active_ineq_row_indices, zeta_sol):
        """
        Pick out the active constraints on the continuous variables as
        A_act * r = b_act
        @param active_ineq_row_indices A set of indices for the active
        inequality constraints.
        @param zeta_sol The solution to the binary variables. A torch array of
        0/1.
        @return (A_act, b_act)
        """
        assert(isinstance(active_ineq_row_indices, set))
        assert(isinstance(zeta_sol, torch.Tensor))
        A_act = torch.zeros(
            (len(self.rhs_eq) + len(active_ineq_row_indices), len(self.r)),
            dtype=self.dtype)
        b_act = torch.zeros(
            (len(self.rhs_eq) + len(active_ineq_row_indices),),
            dtype=self.dtype)
        # First fill in the equality constraints
        # The equality constraints are Aeq_r * r + Aeq_zeta * zeta_sol = beq,
        # equivalent to Aeq_r * r = beq - Aeq_zeta * zeta_sol
        for row, col, val in zip(self.Aeq_r_row, self.Aeq_r_col,
                                 self.Aeq_r_val):
            A_act[row, col] = val
        for i in range(len(self.rhs_eq)):
            b_act[i] = self.rhs_eq[i]
        for row, col, val in zip(self.Aeq_zeta_row, self.Aeq_zeta_col,
                                 self.Aeq_zeta_val):
            b_act[row] -= val * zeta_sol[col]

        # Now fill in the active inequality constraints
        (Ain_r, Ain_zeta, _) = self.get_inequality_constraints()
        for (i, row) in enumerate(active_ineq_row_indices):
            A_act[len(self.rhs_eq) + i] = Ain_r[row]
            b_act[len(self.rhs_eq) + i] = self.rhs_in[row] -\
                Ain_zeta[row] @ zeta_sol
        return (A_act, b_act)

    def get_inequality_constraints(self):
        """
        Return the matrices Ain_r, Ain_zeta, rhs_in as torch tensors.
        """
        Ain_r = torch.zeros((len(self.rhs_in), len(self.r)), dtype=self.dtype)
        Ain_zeta = torch.zeros((len(self.rhs_in), len(self.zeta)),
                               dtype=self.dtype)
        rhs_in = torch.tensor(self.rhs_in)
        for row, col, val in zip(self.Ain_r_row, self.Ain_r_col,
                                 self.Ain_r_val):
            Ain_r[row, col] = val
        for row, col, val in zip(self.Ain_zeta_row, self.Ain_zeta_col,
                                 self.Ain_zeta_val):
            Ain_zeta[row, col] = val
        return (Ain_r, Ain_zeta, rhs_in)


class GurobiTorchMILP(GurobiTorchMIP):
    """
    This class is a subclass of GurobiTorchMIP. It only allows linear cost. The
    MILP is in this form
    min/max cᵣᵀ * r + c_zetaᵀ * ζ + c_constant
    s.t Ain_r * r + Ain_zeta * ζ <= rhs_in
        Aeq_r * r + Aeq_zeta * ζ = rhs_eq
    """

    def __init__(self, dtype):
        GurobiTorchMIP.__init__(self, dtype)
        self.c_r = None
        self.c_zeta = None
        self.c_constant = None
        # Whether the objective is minimization or maximization.
        self.sense = None

    def setObjective(self, coeffs, variables, constant, sense):
        """
        Set the linear objective.
        The objective is ∑ᵢ coeffs[i]ᵀ * variables[i] + constant
        @param coeffs A list of 1D pytorch tensors. coeffs[i] are the
        coefficients for variables[i]
        @param variables A list of lists. variables[i] is a list of gurobi
        variables. Note that the variables cannot overlap.
        @param sense GRB.MAXIMIZE or GRB.MINIMIZE
        """
        # r_used_flag[i] records if r[i] has appeared in @p variables.
        assert(isinstance(coeffs, list))
        assert(isinstance(variables, list))
        assert(len(coeffs) == len(variables))
        assert(sense == gurobipy.GRB.MAXIMIZE or
               sense == gurobipy.GRB.MINIMIZE)
        self.sense = sense
        r_used_flag = [False] * len(self.r)
        zeta_used_flag = [False] * len(self.zeta)
        self.c_r = torch.zeros((len(self.r),), dtype=self.dtype)
        self.c_zeta = torch.zeros((len(self.zeta),), dtype=self.dtype)
        for coeff, var in zip(coeffs, variables):
            assert(isinstance(coeff, torch.Tensor))
            for i in range(len(var)):
                if var[i] in self.r_indices.keys():
                    r_index = self.r_indices[var[i]]
                    if r_used_flag[r_index]:
                        raise Exception("setObjective: variable " + var[i].name
                                        + " is duplicated.")
                    r_used_flag[r_index] = True
                    self.c_r[r_index] = coeff[i]
                elif var[i] in self.zeta_indices.keys():
                    zeta_index = self.zeta_indices[var[i]]
                    if zeta_used_flag[zeta_index]:
                        raise Exception("setObjective: variable " + var[i].name
                                        + " is duplicated.")
                    zeta_used_flag[zeta_index] = True
                    self.c_zeta[zeta_index] = coeff[i]
        if isinstance(constant, float):
            self.c_constant = torch.tensor(constant, dtype=self.dtype)
        elif isinstance(constant, torch.Tensor):
            assert(len(constant.shape) == 0)
            self.c_constant = constant
        else:
            raise Exception("setObjective: constant must be either a float" +
                            " or a torch tensor.")
        self.gurobi_model.setObjective(
            gurobipy.LinExpr(self.c_r, self.r) +
            gurobipy.LinExpr(self.c_zeta, self.zeta) + self.c_constant,
            sense=sense)

    def compute_objective_from_mip_data(
            self, active_ineq_row_indices, zeta_sol):
        """
        Given the active inequality constraints and the value for binary
        variables, compute the objective as a function of the MIP constraint
        / objective data.
        cᵣᵀ * A_act⁻¹ * b_act + c_zetaᵀ * ζ + c_constant
        where A_act, b_act are computed from get_active_constraints
        @return objective cᵣᵀ * A_act⁻¹ * b_act + c_zetaᵀ * ζ + c_constant
        """
        (A_act, b_act) = self.get_active_constraints(
            active_ineq_row_indices, zeta_sol)
        # Now compute A_act⁻¹ * b_act. A_act may not be invertible, so we
        # use its pseudo-inverse (A_actᵀ * A_act)⁻¹ * A_actᵀ * b_act
        return self.c_r @ torch.inverse(A_act.T @ A_act) @ (A_act.T) @ b_act\
            + self.c_zeta @ zeta_sol + self.c_constant

    def compute_objective_from_mip_data_and_solution(
            self, solution_number=0, active_constraint_tolerance=1e-6):
        """
        Suppose the MILP is solved to optimality. We then retrieve the active
        constraints from the (suboptimal) solution, together with the binary
        variable solutions. We can then compute the objective as a function of
        MIP constraint/objective data, by using
        compute_objective_from_mip_data() function.
        @param solution_number The index of the suboptimal solution. Should be
        in the range of [0, gurobi_model.solCount). Setting solution_number to
        0 means the optimal solution.
        @param active_constraint_tolerance If the constraint violation is less
        than this tolerance at the solution, then we think this constraint is
        active at the solution.
        """
        assert(solution_number >= 0 and
               solution_number < self.gurobi_model.solCount)
        assert(self.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL)
        self.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber,
                                   solution_number)
        r_sol = torch.tensor([var.xn for var in self.r], dtype=self.dtype)
        zeta_sol = torch.tensor([var.xn for var in self.zeta],
                                dtype=self.dtype)
        (Ain_r, Ain_zeta, rhs_in) = self.get_inequality_constraints()
        with torch.no_grad():
            lhs_in = Ain_r @ r_sol + Ain_zeta @ zeta_sol
        active_ineq_row_indices = np.arange(len(self.rhs_in))
        active_ineq_row_indices = set(active_ineq_row_indices[
            np.abs(lhs_in.detach().numpy() - rhs_in.detach().numpy()) <
            active_constraint_tolerance])
        return self.compute_objective_from_mip_data(
            active_ineq_row_indices, zeta_sol)
