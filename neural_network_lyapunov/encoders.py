import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        Parent class for encoders. Encoders should be child classes of this
        class and implement the forward method (which takes a tensor
        and returns tensors - see torch documentation).
        Specifically, the forward method needs to return two tensors. The
        first one is mean, and the second the log of the variance of the
        samples in latent spaces (as in most VAE architectues)
        @param z_dim int dimension of the latent space (output)
        @param image_width/height int dimensions of the images (input)
        @param grayscale boolean whether or not the images are in grayscale
        (i.e. have one channel). Otherwise they are RGB (3 channels)
        """
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.image_width = image_width
        self.image_height = image_height
        self.grayscale = grayscale
        if self.grayscale:
            self.num_channels_in = 2
        else:
            self.num_channels_in = 6

    def layers_output_shape(self, layers):
        """
        compute the output size of a list of layers, given an input image
        @param conv list of torch layers (like nn.Conv2d)
        @return tuple (num_channels, width, height)
        """
        with torch.no_grad():
            x_tmp = torch.rand(
                (1, self.num_channels_in, self.image_width, self.image_height))
            for c_layer in self.conv:
                x_tmp = c_layer.forward(x_tmp)
        return x_tmp.shape[1:]


class Decoder(nn.Module):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        Parent class for decoders. Decoders should be child classes of this
        class and implement the forward method (which takes a tensor
        and returns a tensor - see torch documentation).
        @param z_dim int dimension of the latent space (input)
        @param image_width/height int dimensions of the images (output)
        @param grayscale boolean whether or not the images are in grayscale
        (i.e. have one channel). Otherwise they are RGB (3 channels)
        """
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.image_width = image_width
        self.image_height = image_height
        self.grayscale = grayscale
        if grayscale:
            self.num_channels_out = 2
        else:
            self.num_channels_out = 6


class LinearEncoder1(Encoder):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        See Encoder documentation
        """
        super(LinearEncoder1, self).__init__(z_dim, image_width, image_height,
                                             grayscale)
        linear = [
            nn.Linear(
                self.num_channels_in * self.image_width * self.image_height,
                500),
            nn.Linear(500, 500),
            nn.Linear(500, self.z_dim * 2),
        ]
        self.linear = nn.ModuleList(linear)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        returns mean and log of variance in latent space as two tensors
        """
        x = torch.flatten(x, start_dim=1)
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        return x[:, :self.z_dim], x[:, self.z_dim:]


class LinearDecoder1(Decoder):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        See Decoder documentation
        """
        super(LinearDecoder1, self).__init__(z_dim, image_width, image_height,
                                             grayscale)
        linear = [
            nn.Linear(self.z_dim, 500),
            nn.Linear(500, 500),
            nn.Linear(
                500,
                self.num_channels_out * self.image_width * self.image_height),
        ]
        self.linear = nn.ModuleList(linear)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        x = x.view(x.shape[0], self.num_channels_out, self.image_width,
                   self.image_height)
        x = self.sigmoid(x)
        return x


class CNNEncoder1(Encoder):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        See Encoder documentation
        """
        super(CNNEncoder1, self).__init__(z_dim, image_width, image_height,
                                          grayscale)
        conv = [
            nn.Conv2d(self.num_channels_in, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.MaxPool2d(2, 2),
        ]
        self.conv = nn.ModuleList(conv)
        conv_out_shape = self.layers_output_shape(self.conv)
        linear = [
            nn.Linear(
                conv_out_shape[0] * conv_out_shape[1] * conv_out_shape[2],
                self.z_dim * 2),
        ]
        self.linear = nn.ModuleList(linear)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        returns mean and log of variance in latent space as two tensors
        """
        for c_layer in self.conv:
            x = self.relu(c_layer(x))
        x = torch.flatten(x, start_dim=1)
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        return x[:, :self.z_dim], x[:, self.z_dim:]


class CNNDecoder1(Decoder):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        See Decoder documentation
        """
        super(CNNDecoder1, self).__init__(z_dim, image_width, image_height,
                                          grayscale)
        width_in = int(self.image_width / 4)
        height_in = int(self.image_height / 4)
        self.height_in = height_in
        self.width_in = width_in
        linear = [
            nn.Linear(self.z_dim, self.width_in * self.height_in),
        ]
        self.linear = nn.ModuleList(linear)
        conv = [
            nn.ConvTranspose2d(1, 16, 2, stride=2),
            nn.ConvTranspose2d(16, self.num_channels_out, 2, stride=2),
        ]
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        x = x.view(x.shape[0], 1, self.width_in, self.height_in)
        for c_layer in self.conv[:-1]:
            x = self.relu(c_layer(x))
        x = self.conv[-1](x)
        x = self.sigmoid(x)
        return x


class CNNEncoder2(Encoder):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        See Encoder documentation
        """
        super(CNNEncoder2, self).__init__(z_dim, image_width, image_height,
                                          grayscale)
        conv = [
            nn.Conv2d(self.num_channels_in, 32, 5, stride=1, padding=0),
            nn.Conv2d(32, 32, 5, stride=2, padding=0),
            nn.Conv2d(32, 32, 5, stride=2, padding=0),
            nn.Conv2d(32, 10, 5, stride=2, padding=0),
        ]
        self.conv = nn.ModuleList(conv)
        conv_out_shape = self.layers_output_shape(self.conv)
        linear = [
            nn.Linear(
                conv_out_shape[0] * conv_out_shape[1] * conv_out_shape[2],
                500),
            nn.Linear(500, self.z_dim * 2),
        ]
        self.linear = nn.ModuleList(linear)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        returns mean and log of variance in latent space as two tensors
        """
        for c_layer in self.conv:
            x = self.relu(c_layer(x))
        x = torch.flatten(x, start_dim=1)
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        return x[:, :self.z_dim], x[:, self.z_dim:]


class CNNDecoder2(Decoder):
    def __init__(self, z_dim, image_width, image_height, grayscale):
        """
        See Decoder documentation
        """
        super(CNNDecoder2, self).__init__(z_dim, image_width, image_height,
                                          grayscale)
        width_in = self.image_width - 4
        height_in = self.image_height - 4
        for k in range(2):
            width_in = int(width_in / 2)
            height_in = int(height_in / 2)
            width_in = int(width_in - 4)
            height_in = int(height_in - 4)
        self.height_in = height_in
        self.width_in = width_in
        linear = [
            nn.Linear(self.z_dim, 200),
            nn.Linear(200, 1000),
            nn.Linear(1000, self.width_in * self.height_in),
        ]
        self.linear = nn.ModuleList(linear)
        conv = [
            nn.ConvTranspose2d(1, 32, 5, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32,
                               self.num_channels_out,
                               5,
                               stride=1,
                               padding=0),
        ]
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for l_layer in self.linear[:-1]:
            x = self.relu(l_layer(x))
        x = self.linear[-1](x)
        x = x.view(x.shape[0], 1, self.width_in, self.height_in)
        for c_layer in self.conv[:-1]:
            x = self.relu(c_layer(x))
        x = self.conv[-1](x)
        x = self.sigmoid(x)
        return x
