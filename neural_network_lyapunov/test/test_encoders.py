import unittest
import torch

import neural_network_lyapunov.encoders as encoders


class TestEncoders(unittest.TestCase):

    def setUp(self):
        z_dim = 5
        image_width = 48
        image_height = 48
        self.pairs = [
            (encoders.LinearEncoder1(z_dim, image_width, image_height, True),
             encoders.LinearDecoder1(z_dim, image_width, image_height, True)),
            (encoders.LinearEncoder1(z_dim, image_width, image_height, False),
             encoders.LinearDecoder1(z_dim, image_width, image_height, False)),
            (encoders.CNNEncoder1(z_dim, image_width, image_height, True),
             encoders.CNNDecoder1(z_dim, image_width, image_height, True)),
            (encoders.CNNEncoder1(z_dim, image_width, image_height, False),
             encoders.CNNDecoder1(z_dim, image_width, image_height, False)),
            (encoders.CNNEncoder2(z_dim, image_width, image_height, True),
             encoders.CNNDecoder2(z_dim, image_width, image_height, True)),
            (encoders.CNNEncoder2(z_dim, image_width, image_height, False),
             encoders.CNNDecoder2(z_dim, image_width, image_height, False)),
        ]

    def test_encoder_decoder_pairs(self):
        def check_pair(enc, dec):
            x_in = torch.rand((3, enc.num_channels_in,
                              enc.image_width, enc.image_height))
            # latent dim is OK
            z = enc(x_in)
            self.assertEqual(len(z), 2)
            self.assertTrue(isinstance(z, tuple))
            self.assertEqual(z[0].shape, z[1].shape)
            self.assertEqual(z[0].shape, (3, enc.z_dim))
            # input shape = output shape
            x_out = dec(z[0])
            self.assertEqual(x_out.shape, x_in.shape)
            # all output 0 < x < 1
            self.assertTrue(torch.all(x_out >= 0))
            self.assertTrue(torch.all(x_out <= 1))
        for pair in self.pairs:
            check_pair(pair[0], pair[1])


if __name__ == "__main__":
    unittest.main()
