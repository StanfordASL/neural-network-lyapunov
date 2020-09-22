import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter


class TrainingLog():
    def __init__(self, loss_dim, prefix="", writer=None,
                 first_value_only=False):
        self.loss_dim = loss_dim
        self.prefix = prefix
        self.first_value_only = first_value_only
        if writer is None:
            self.writer = SummaryWriter()
        else:
            self.writer = writer
        self.train_losses = np.zeros((0, loss_dim))
        self.validation_losses = np.zeros((0, loss_dim))

    def add_train_loss(self, losses):
        self.train_losses = np.vstack((self.train_losses, losses))
        if self.first_value_only:
            self.writer.add_scalars('Train',
                                    {self.prefix: losses[0].item()},
                                    self.train_losses.shape[0])
        else:
            for i in range(self.loss_dim):
                self.writer.add_scalars('Train_' + str(i),
                                        {self.prefix: losses[i].item()},
                                        self.train_losses.shape[0])

    def add_validation_loss(self, losses):
        self.validation_losses = np.vstack((self.validation_losses, losses))
        if self.first_value_only:
            self.writer.add_scalars('Validation',
                                    {self.prefix: losses[0].item()},
                                    self.train_losses.shape[0])
        else:
            for i in range(self.loss_dim):
                self.writer.add_scalars('Validation_' + str(i),
                                        {self.prefix: losses[i].item()},
                                        self.train_losses.shape[0])

    @staticmethod
    def get_copy(log, prefix=None, first_value_only=None, keep_writer=False):
        if prefix is None:
            prefix = log.prefix
        if first_value_only is None:
            first_value_only = log.first_value_only
        if keep_writer:
            writer = log.writer
        else:
            writer = None
        log_cp = TrainingLog(log.loss_dim, prefix=prefix, writer=writer,
                             first_value_only=first_value_only)
        log_cp.train_losses = copy.deepcopy(log.train_losses)
        log_cp.validation_losses = copy.deepcopy(log.validation_losses)
        return log_cp
