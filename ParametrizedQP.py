import torch

class ParametrizedQP:
    def __init__(self, device, dtype=torch.DoubleTensor):
        self.device = device
        self.dtype = dtype
        self.Q = torch.zeros(0,device=self.device).type(self.dtype)
        self.q = torch.zeros(0,device=self.device).type(self.dtype)
        self.G = torch.zeros(0,device=self.device).type(self.dtype)
        self.h = torch.zeros(0,device=self.device).type(self.dtype)
        self.A = torch.zeros(0,device=self.device).type(self.dtype)
        self.b = torch.zeros(0,device=self.device).type(self.dtype)
        self.var_selector = {}
        self.next_index = 0
    
    def add_var(self,name,size):
        self.var_selector[name] = (self.next_index,self.next_index+size)
        self.G = torch.cat((self.G,torch.zeros(self.G.shape[0],size,device=self.device).type(self.dtype)),1)
        self.A = torch.cat((self.A,torch.zeros(self.A.shape[0],size,device=self.device).type(self.dtype)),1)
        self.Q = torch.cat((self.Q,torch.zeros(self.Q.shape[0],size,device=self.device).type(self.dtype)),1)
        self.Q = torch.cat((self.Q,torch.zeros(size,self.Q.shape[1],device=self.device).type(self.dtype)),0)
        self.q = torch.cat((self.q,torch.zeros(size,device=self.device).type(self.dtype)))
        self.next_index += size
    
    @property
    def num_vars(self):
        return self.next_index
        
    def _build_con(self,names,coeffs):
        if len(coeffs[0].shape) > 1:
            size = coeffs[0].shape[0]
        else:
            size = 1
        con = torch.zeros(size, self.num_vars, device=self.device).type(self.dtype)
        for i in range(len(names)):
            var_range = self.var_selector[names[i]] 
            con[:,var_range[0]:var_range[1]] = coeffs[i]
        return con
        
    def _convert_type(self,x):
        if not isinstance(x,self.dtype):
            x = self.dtype(x)
        x.to(self.device)
        return x
        
    def add_ineq(self,names,coeffs,bound):
        coeffs = [self._convert_type(c) for c in coeffs]
        bound = self._convert_type(bound)
        con = self._build_con(names,coeffs)
        self.G = torch.cat((self.G, con), 0)
        self.h = torch.cat((self.h, bound), 0)
        
    def add_eq(self,names,coeffs,bound):
        coeffs = [self._convert_type(c) for c in coeffs]
        bound = self._convert_type(bound)
        con = self._build_con(names,coeffs)
        self.A = torch.cat((self.A, con), 0)
        self.b = torch.cat((self.b, bound), 0)   
        
    def set_obj(self,name,quadratic=None,linear=None):
        var_range = self.var_selector[name]
        if type(quadratic) != type(None):
            self.Q[var_range[0]:var_range[1],var_range[0]:var_range[1]] = self._convert_type(quadratic)
        if type(linear) != type(None):
            self.q[var_range[0]:var_range[1]] = self._convert_type(linear)
            
    def eval_obj(self,x):
        return .5 * x @ self.Q @ x.T + self.q @ x.T 