import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter


class TrainingLog():
    def __init__(self, prefix="", writer=None):
        self.prefix = prefix
        if writer is None:
            self.writer = SummaryWriter()
        else:
            self.writer = writer
        self.train_losses = np.zeros((0, 1))
        self.validation_losses = np.zeros((0, 1))

    def add_train_loss(self, loss):
        self.train_losses = np.vstack((self.train_losses, loss))
        self.writer.add_scalars('Train',
                                {self.prefix: loss},
                                 self.train_losses.shape[0])

    def add_validation_loss(self, loss):
        self.validation_losses = np.vstack((self.validation_losses, loss))
        self.writer.add_scalars('Validation',
                                {self.prefix: loss},
                                 self.train_losses.shape[0])

    @staticmethod
    def get_copy(log, prefix=None, keep_writer=False):
        if prefix is None:
            prefix = log.prefix
        if keep_writer:
            writer = log.writer
        else:
            writer = None
        log_cp = TrainingLog(prefix=prefix, writer=writer)
        log_cp.train_losses = copy.deepcopy(log.train_losses)
        log_cp.validation_losses = copy.deepcopy(log.validation_losses)
        return log_cp
