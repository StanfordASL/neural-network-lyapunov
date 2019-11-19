import gurobipy
import torch


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
        assert(isinstance(rhs, torch.Tensor))
        expr = 0
        assert(isinstance(coeffs, list))
        assert(len(coeffs) == len(variables))
        num_vars = 0
        for coeff, var in zip(coeffs, variables):
            assert(isinstance(coeff, torch.Tensor))
            expr += gurobipy.LinExpr(coeff.tolist(), var)
            num_vars += len(var)
        constr = self.gurobi_model.addLConstr(expr, sense=sense, rhs=rhs,
                                              name=name)
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
            self.rhs_eq.append(rhs)
        else:
            if num_cont_vars > 0:
                self.Ain_r_row.extend(new_Ain_r_row[:num_cont_vars])
                self.Ain_r_col.extend(new_Ain_r_col[:num_cont_vars])
                self.Ain_r_val.extend(new_Ain_r_val[:num_cont_vars])
            if num_bin_vars > 0:
                self.Ain_zeta_row.extend(new_Ain_zeta_row[:num_bin_vars])
                self.Ain_zeta_col.extend(new_Ain_zeta_col[:num_bin_vars])
                self.Ain_zeta_val.extend(new_Ain_zeta_val[:num_bin_vars])
            self.rhs_in.append(rhs if sense == gurobipy.GRB.LESS_EQUAL else
                               -rhs)

        return constr


class GurobiTorchMILP(GurobiTorchMIP):
    """
    This class is a subclass of GurobiTorchMIP. It only allows linear cost. The
    MILP is in this form
    min cᵣᵀ * r + c_zetaᵀ * ζ + c_constant
    s.t Ain_r * r + Ain_zeta * ζ <= rhs_in
        Aeq_r * r + Aeq_zeta * ζ = rhs_eq
    """

    def __init__(self, dtype):
        GurobiTorchMIP.__init__(self, dtype)
        self.c_r = None
        self.c_zeta = None
        self.c_constant = None

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
                    self.c_r[r_index] = coeff[i] if\
                        sense == gurobipy.GRB.MINIMIZE else -coeff[i]
                elif var[i] in self.zeta_indices.keys():
                    zeta_index = self.zeta_indices[var[i]]
                    if zeta_used_flag[zeta_index]:
                        raise Exception("setObjective: variable " + var[i].name
                                        + " is duplicated.")
                    zeta_used_flag[zeta_index] = True
                    self.c_zeta[zeta_index] = coeff[i] if\
                        sense == gurobipy.GRB.MINIMIZE else -coeff[i]
        if isinstance(constant, float):
            self.c_constant = torch.tensor(constant, dtype=self.dtype) if\
                sense == gurobipy.GRB.MINIMIZE else\
                torch.tensor(-constant, dtype=self.dtype)
        elif isinstance(constant, torch.Tensor):
            assert(len(constant.shape) == 0)
            self.c_constant = constant if sense == gurobipy.GRB.MINIMIZE else\
                -constant
        else:
            raise Exception("setObjective: constant must be either a float" +
                            " or a torch tensor.")
