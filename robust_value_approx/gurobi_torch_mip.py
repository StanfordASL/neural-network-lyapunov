import gurobipy
import torch

class GurobiTorchMIP:
    """
    This class will be used in computing the gradient of an MILP optimal cost
    w.r.t constraint/objective data. It uses gurobi to solve the MILP, but also
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
        self.dtype=dtype
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
        new_vars = self.gurobi_model.addVars(
            num_vars, lb=lb, vtype=vtype, name=name)
        self.gurobi_model.update()
        if vtype==gurobipy.GRB.CONTINUOUS:
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
        elif vtype==gurobipy.GRB.BINARY:
            num_existing_zeta = len(self.zeta_indices)
            self.zeta.extend([new_vars[i] for i in range(num_vars)])
            for i in range(num_vars):
                self.zeta_indices[new_vars[i]] = num_existing_zeta + i
        else:
            raise Exception("Only support continuous or binary variables")
        return new_vars
