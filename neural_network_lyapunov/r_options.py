import torch
import numpy as np


class ROptions:
    """
    When search for the Lyapunov function, we use the 1-norm of |R*(x-x*)|₁.
    This class specificies the options to search for R.
    """
    def __init__(self):
        pass

    def set_variable_value(self, R_val: np.ndarray):
        pass

    def R(self) -> torch.Tensor:
        pass

    def variables(self) -> list:
        pass

    @property
    def fixed_R(self) -> bool:
        pass

    def extract_params(self):
        return dict()


class SearchRwithSPDOptions(ROptions):
    """
    We want R to be a full column rank matrix, with size m x n and m >= n.
    The first n rows of R (The square matrix on top of R) is parameterized
    as L * L' + eps * I to make sure the induced 2-norm of R is at least
    eps. Namely the top square matrix of R is a symmetric positive definite
    (SPD) matrix.
    """
    def __init__(self, R_size, epsilon):
        """
        We want R to be a full column rank matrix, with size m x n and m >= n.
        The first n rows of R (The square matrix on top of R) is parameterized
        as L * L' + eps * I to make sure the induced 2-norm of R is at least
        eps.
        @param R_size the size of R, with R_size[0] >= R_size[1]
        @param epsilon eps in the documentation above.
        """
        super(SearchRwithSPDOptions, self).__init__()
        assert (len(R_size) == 2)
        assert (R_size[0] >= R_size[1])
        self.R_size = R_size
        self._variables = torch.empty((int(R_size[1] * (R_size[1] + 1) / 2) +
                                       (R_size[0] - R_size[1]) * R_size[1], ),
                                      dtype=torch.float64,
                                      requires_grad=True)
        assert (epsilon > 0)
        self.epsilon = epsilon

    def set_variable_value(self, R_val: np.ndarray):
        assert (isinstance(R_val, np.ndarray))
        assert (R_val.shape == self.R_size)
        R_top = R_val[:R_val.shape[1], :]
        R_top = (R_top + R_top.T) / 2
        L = np.linalg.cholesky(R_top - self.epsilon * np.eye(R_val.shape[1]))
        L_entry_count = 0
        variable_val = np.empty((self._variables.shape[0], ))
        for i in range(self.R_size[1]):
            variable_val[L_entry_count:L_entry_count+self.R_size[1]-i] =\
                L[i:, i]
            L_entry_count += self.R_size[1] - i
        variable_val[L_entry_count:] = R_val[self.R_size[1]:, :].reshape(
            (-1, ))
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def set_variable_value_directly(self, variable_val: np.ndarray):
        assert (isinstance(variable_val, np.ndarray))
        assert (variable_val.shape == self._variables.shape)
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def R(self):
        L_entry_count = int(self.R_size[1] * (self.R_size[1] + 1) / 2)
        L_lower_list = torch.split(
            self._variables[:L_entry_count],
            np.arange(1, self.R_size[1] + 1, 1, dtype=int)[::-1].tolist())
        L_list = []
        for i in range(self.R_size[1]):
            L_list.append(torch.zeros((i, ), dtype=torch.float64))
            L_list.append(L_lower_list[i])
        L = torch.cat(L_list).reshape((self.R_size[1], self.R_size[1])).T
        R_bottom = self._variables[L_entry_count:].reshape(
            (self.R_size[0] - self.R_size[1], self.R_size[1]))
        R = torch.cat(
            (L @ L.T +
             self.epsilon * torch.eye(self.R_size[1], dtype=torch.float64),
             R_bottom),
            dim=0)
        return R

    def variables(self):
        return [self._variables]

    def __str__(self):
        return f"Search R with SPD, size {self.R_size} and epsilon" +\
            f" {self.epsilon}"

    @property
    def fixed_R(self):
        return False

    def extract_params(self):
        return {
            "R_size": self.R_size,
            "R_epsilon": self.R_epsilon,
            "R_variables": self.R_variables()
        }


class FixedROptions(ROptions):
    """
    When search for the Lyapunov function, we use the 1-norm of |R*(x-x*)|₁.
    This class specificies that R is fixed.
    R should be fixed to a full column rank matrix.
    """
    def __init__(self, R: torch.Tensor):
        super(FixedROptions, self).__init__()
        assert (isinstance(R, torch.Tensor))
        self._R = R

    def R(self):
        return self._R

    def variables(self):
        return []

    def __str__(self):
        return f"Fixed R to \n {self._R}"

    @property
    def fixed_R(self):
        return True


class SearchRwithSVDOptions(ROptions):
    """
    Search R by searching its singular value. R = U * Σ * V, where U and V are
    given orthonormal matrices, Σ is a diagonal matrix whose diagonal entries
    are the singular values. We parameterize Σ[i, i] = a(i) + v(i)², where v(i)
    is the variable we search for, and a(i) is a given positive scalar. This
    guarantees that the minimal singular value of R is larger than min(a).
    """
    def __init__(self, R_size: tuple, a: np.ndarray):
        """
        R = U * Σ * V, where U and V are given orthornomral  matrices. Σ is a
        diagonal matrix such that Σ[i, i] = a(i) + v(i)²
        """
        super(SearchRwithSVDOptions, self).__init__()
        self.R_size = R_size
        assert (isinstance(a, np.ndarray))
        assert (a.shape == (np.min(self.R_size), ))
        assert (np.all(a > 0))
        self.a = a
        self._variables = torch.empty((a.shape[0], ),
                                      dtype=torch.float64,
                                      requires_grad=True)
        self.U = np.eye(self.R_size[0])
        self.V = np.eye(self.R_size[1])

    def set_variable_value(self, R_val: np.ndarray):
        assert (isinstance(R_val, np.ndarray))
        assert (R_val.shape == self.R_size)
        self.U, Sigma, self.V = np.linalg.svd(R_val)
        assert (np.all(Sigma >= self.a))
        variable_val = np.sqrt(Sigma - self.a)
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def set_variable_value_directly(self, variable_val: np.ndarray):
        assert (isinstance(variable_val, np.ndarray))
        assert (variable_val.shape == (np.min(self.R_size), ))
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def R(self):
        Sigma_diag = torch.diag(torch.from_numpy(self.a) + self._variables**2)
        if self.R_size[0] > self.R_size[1]:
            Sigma = torch.cat(
                (Sigma_diag,
                 torch.zeros((self.R_size[0] - self.R_size[1], self.R_size[1]),
                             dtype=torch.float64)),
                dim=0)
        elif self.R_size[0] < self.R_size[1]:
            Sigma = torch.cat(
                (Sigma_diag,
                 torch.zeros((self.R_size[0], self.R_size[1] - self.R_size[0]),
                             dtype=torch.float64)),
                dim=1)
        else:
            Sigma = Sigma_diag
        return torch.from_numpy(self.U) @ Sigma @ torch.from_numpy(self.V)

    def variables(self) -> list:
        return [self._variables]

    def __str__(self):
        return f"Search R with SVD. Size {self.R_size}\na: {self.a}\n" + \
            f"U: {self.U}\nV:{self.V}"

    @property
    def fixed_R(self):
        return False

    def extract_params(self):
        return {
            "R_size": self.R_size,
            "R_U": self.U,
            "R_V": self.V,
            "R_a": self.a
        }


class SearchRfreeOptions(ROptions):
    def __init__(self, R_size: tuple):
        super(SearchRfreeOptions, self).__init__()
        assert (isinstance(R_size, tuple))
        assert (len(R_size) == 2)
        self.R_size = R_size
        self._variables = torch.empty(R_size,
                                      dtype=torch.float64,
                                      requires_grad=True)

    def set_variable_value(self, R_val: np.ndarray):
        assert (isinstance(R_val, np.ndarray))
        assert (R_val.shape == self.R_size)
        self._variables = torch.from_numpy(R_val)
        self._variables.requires_grad = True

    def set_variable_value_directly(self, variable_val: np.ndarray):
        assert (isinstance(variable_val, np.ndarray))
        assert (variable_val.shape == self.R_size)
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def R(self):
        return self._variables

    def variables(self) -> list:
        return [self._variables]

    def __str__(self):
        return f"Search R freely. Size {self.R_size}"

    @property
    def fixed_R(self):
        return False

    def extract_params(self):
        return {"R_size": self.R_size}
