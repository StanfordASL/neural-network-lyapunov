import torch
import numpy as np


class SamplesBuffer:
    def __init__(self, samples_dim, labels_dim, dtype, max_size=None):
        self.samples_dim = samples_dim
        self.dtype = dtype
        self.max_size = max_size
        self.x_samples = torch.Tensor(0, samples_dim).type(dtype)
        self.v_labels = torch.Tensor(0, labels_dim).type(dtype)

    def add_samples(self, new_x_samples, new_v_labels):
        assert(new_x_samples.shape[0] == new_v_labels.shape[0])
        num_new_samples = new_x_samples.shape[0]
        if (self.max_size is not None and
                self.num_samples + num_new_samples > self.max_size):
            num_extra = self.max_size - num_new_samples
            self.x_samples = self.x_samples[num_extra:, :]
            self.v_labels = self.v_labels[num_extra:, :]
        self.x_samples = torch.cat((self.x_samples, new_x_samples), axis=0)
        self.v_labels = torch.cat((self.v_labels, new_v_labels), axis=0)

    def get_random_sample_indices(self, num_indices):
        indices = np.random.choice(self.num_samples,
                                   min(self.num_samples, num_indices),
                                   replace=False)
        return indices

    def get_samples_from_indices(self, indices):
        return(self.x_samples[indices, :], self.v_labels[indices, :])

    def get_random_samples(self, num_rand_samples):
        indices = self.get_random_sample_indices(num_rand_samples)
        return self.get_samples_from_indices(indices)

    @property
    def num_samples(self):
        return self.x_samples.shape[0]
