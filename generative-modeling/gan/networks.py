import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    # 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.upscale_factor = upscale_factor
        self.pix_shuffle    = nn.PixelShuffle(self.upscale_factor)
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size = (kernel_size, kernel_size), padding = (padding, padding))

    @torch.jit.script_method
    def forward(self, x):
        # 1.1: Implement nearest neighbor upsampling
   
        # 1. Repeat x channel wise upscale_factor^2 times
        x_repeat = x.repeat(1, int(self.upscale_factor**2), 1, 1)

        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        x_up = self.pix_shuffle(x_repeat)

        # 3. Apply convolution and return output
        return self.conv(x_up)


class DownSampleConv2D(torch.jit.ScriptModule):
    # 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.downscale_factor = downscale_ratio
        self.pix_unshuffle    = nn.PixelUnshuffle(self.downscale_factor)
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), padding = (padding, padding))


    @torch.jit.script_method
    def forward(self, x):
        # 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        x_down = self.pix_unshuffle(x)
        N, C, H, W = x_down.shape
        # print(f"N:{N} C:{C} H:{H} W:{W}")
        
        # 2. Then split channel wise into (downscale_factor^2 x batch x channel x height x width) images
        C_new = int(C // self.downscale_factor**2)
        # I feel better about doing below instead of what was suggested, I don't think it really matters tbh, though
        x_down_rsz = x_down.view(N, int(self.downscale_factor**2), C_new, H, W)

        # 3. Average across dimension 0, apply convolution and return output
        return self.conv(torch.mean(x_down_rsz, dim = 1)).reshape(N, 128, H, W)


class ResBlockUp(torch.jit.ScriptModule):
    # 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        # 1.1: Setup the network layers
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(n_filters, kernel_size, n_filters, padding = 1)
        )

        self.upsample_residual = UpSampleConv2D(input_channels, n_filters = n_filters, kernel_size = 1)

    @torch.jit.script_method
    def forward(self, x):
        # 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        x_up   = self.upsample_residual(x)
        x_layer = self.layers(x)
        return x_up + x_layer


class ResBlockDown(torch.jit.ScriptModule):
    # 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # 1.1: Setup the network layers
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), stride = (1,1), padding=(1,1)),
            nn.ReLU(),
            DownSampleConv2D(n_filters, n_filters=n_filters, kernel_size=3, padding=1)
        )

        self.downsample_residual = DownSampleConv2D(input_channels, n_filters=n_filters, kernel_size=1)

    @torch.jit.script_method
    def forward(self, x):
        # 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        layer_out = self.layers(x)
        down_out  = self.downsample_residual(x)
        return layer_out + down_out

class ResBlock(torch.jit.ScriptModule):
    # 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, (kernel_size, kernel_size), stride = (1, 1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, (kernel_size, kernel_size), stride = (1, 1), padding=(1,1))
        )

    @torch.jit.script_method
    def forward(self, x):
        x_layers = self.layers(x)
        return x + x_layers
        
class Generator(torch.jit.ScriptModule):
    # 1.1: Implement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        """
        Instaniate the generator for GAN

        Args:
            starting_image_size (int, optional): 
            Specifies the spatial size of the input to ghe generator's first layer. This is what resizes after the first dense layer Defaults to 4.
        """
        super(Generator, self).__init__()
        # 1.1: Setup the network layers
        self.start_size = starting_image_size

        self.dense  = nn.Linear(in_features=128, out_features=2048, bias=True)
        self.layers = nn.Sequential(
            ResBlockUp(128),
            ResBlockUp(128),
            ResBlockUp(128),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh()
        )

    @torch.jit.script_method
    def forward_given_samples(self, z):
        # 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        out = self.dense(z)
        out = out.view(out.shape[0], int(out.shape[1] // self.start_size**2), self.start_size, self.start_size)
        return self.layers(out)

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # 1.1: Generate n_samples latents and forward through the network.

        # Sample from standard normal distribution
        z = torch.randn((n_samples, 128)).cuda()
        # Below should be the same
        # z = torch.normal(mean = 0., std = 1., size = (n_samples, 128)).cuda()

        # Pass the result through the network
        return self.forward_given_samples(z)


class Discriminator(torch.jit.ScriptModule):
    # 1.1: Implement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # 1.1: Setup the network layers
        self.layers = nn.Sequential(
            ResBlockDown(3), # 3 Channel image
            ResBlockDown(128),
            ResBlock(128),
            ResBlock(128),
            nn.ReLU()
        )

        # Output is between [0,1] with 1 being a fake
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)

    @torch.jit.script_method
    def forward(self, x):
        # 1.1: Forward the discriminator assuming a batch of images have been passed in.
        out = self.layers(x)
        # NOTE: Make sure to sum across the image dimensions after passing x through self.layers.
        sum_out = torch.sum(out, dim = (2, 3))
        return self.dense(sum_out)
        
