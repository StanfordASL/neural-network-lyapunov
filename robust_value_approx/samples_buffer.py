import torch
import numpy as np


class SamplesBuffer:
    def __init__(self, samples_dim, labels_dim, dtype, max_size=None):
        """
        Container for samples from a value function.
        @param samples_dim int length of the data points (e.g. for an infinite
        horizon problem this is x_dim, for a finite one, this would be
        x_dim*(N-1))
        @param label_dim int length of the labels (e.g. for an infinite horizon
        problem this would be 1 but for a finite one it would be N-1)
        @param dtype torch.dtype of the smaples
        @param max_size int maximum size of the buffer in number of samples.
        Past this number the buffer acts as a FIFO queue
        """
        self.samples_dim = samples_dim
        self.dtype = dtype
        self.max_size = max_size
        self.x_samples = torch.Tensor(0, samples_dim).type(dtype)
        self.v_labels = torch.Tensor(0, labels_dim).type(dtype)

    def add_samples(self, new_x_samples, new_v_labels):
        """
        Add samples to the buffer
        @param new_x_samples Tensor n X samples_dim
        @param new_v_labels Tensor n X labels_dim
        """
        assert(new_x_samples.shape[0] == new_v_labels.shape[0])
        if self.max_size is not None:
            assert(new_x_samples.shape[0] <= self.max_size)
        num_new_samples = new_x_samples.shape[0]
        if (self.max_size is not None and
                self.num_samples + num_new_samples > self.max_size):
            num_extra = self.max_size - num_new_samples
            self.x_samples = self.x_samples[num_extra:, :]
            self.v_labels = self.v_labels[num_extra:, :]
        self.x_samples = torch.cat((self.x_samples, new_x_samples), axis=0)
        self.v_labels = torch.cat((self.v_labels, new_v_labels), axis=0)

    def get_random_sample_indices(self, num_indices):
        """
        @return random samples indices that can be used to recover random
        samples
        """
        indices = np.random.choice(self.num_samples,
                                   min(self.num_samples, num_indices),
                                   replace=False)
        return indices

    def get_samples_from_indices(self, indices):
        """
        @return tuple of tensor of the (samples, labels) corresponding to
        indices
        """
        return(self.x_samples[indices, :], self.v_labels[indices, :])

    def get_random_samples(self, num_rand_samples):
        """
        @return a tuple with (random samples, random labels)
        """
        indices = self.get_random_sample_indices(num_rand_samples)
        return self.get_samples_from_indices(indices)

    @property
    def num_samples(self):
        """
        @return number of samples currently in the buffer
        """
        return self.x_samples.shape[0]


class TimedSamplesBuffer:
    def __init__(self, samples_dim, labels_dim, dtype, max_size=None):
        self.samples_dim = samples_dim
        self.dtype = dtype
        self.max_size = max_size
        self.x_samples = torch.Tensor(0, samples_dim).type(dtype)
        self.v_labels = torch.Tensor(0, labels_dim).type(dtype)
        self.t_samples = torch.Tensor(0, labels_dim).type(dtype)

    def add_samples(self, new_x_samples, new_v_labels, new_t_samples):
        assert(new_x_samples.shape[0] == new_v_labels.shape[0])
        assert(new_t_samples.shape[0] == new_v_labels.shape[0])
        if self.max_size is not None:
            assert(new_x_samples.shape[0] <= self.max_size)
        num_new_samples = new_x_samples.shape[0]
        if (self.max_size is not None and
                self.num_samples + num_new_samples > self.max_size):
            num_extra = self.max_size - num_new_samples
            self.x_samples = self.x_samples[num_extra:, :]
            self.v_labels = self.v_labels[num_extra:, :]
            self.t_samples = self.t_samples[num_extra, :]
        self.x_samples = torch.cat((self.x_samples, new_x_samples), axis=0)
        self.v_labels = torch.cat((self.v_labels, new_v_labels), axis=0)
        self.t_samples = torch.cat((self.t_samples, new_t_samples), axis=0)

    def get_random_sample_indices(self, num_indices):
        indices = np.random.choice(self.num_samples,
                                   min(self.num_samples, num_indices),
                                   replace=False)
        return indices

    def get_samples_from_indices(self, indices):
        return(self.x_samples[indices, :], self.v_labels[indices, :], self.t_samples[indices, :])

    def get_random_samples(self, num_rand_samples):
        indices = self.get_random_sample_indices(num_rand_samples)
        return self.get_samples_from_indices(indices)

    @property
    def num_samples(self):
        return self.x_samples.shape[0]
