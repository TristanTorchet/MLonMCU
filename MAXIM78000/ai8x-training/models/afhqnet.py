###################################################################################################
# MemeNet network
# Marco Giordano
# Center for Project Based Learning
# 2022 - ETH Zurich
###################################################################################################
"""
MemeNet network description
"""
from signal import pause
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

"""
Network description class
"""
class AFHQNet(nn.Module):
    """
    7-Layer CNN - Lightweight image classification
    """
    def __init__(self, num_classes=3, dimensions=(64, 64), num_channels=3, bias=False, **kwargs):
        super().__init__()

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        # Images are resized to 64x64 in the memes.py script
        dim_x, dim_y = dimensions

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        # Images are resized to 64x64 in the memes.py script
        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(in_channels=num_channels,
                                          out_channels=3, kernel_size=3,
                                          padding=1, bias=bias, **kwargs)

        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels=3,
                                                 out_channels=8, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # input shape: in_channels x in x in = 32 x in x in
        # pooling: pool_size = 2, pool_stride = 2, pool_dilation = 1
        # output pooling shape: 32 x floor(in/2) x floor(in/2)
        # padding 1, stride 1, kernel 3 -> no change in dimensions
        # output conv shape: out_channels x floor(in/2) x floor(in/2)
        #                   =          24 x floor(in/2) x floor(in/2)

        dim_x //= 2
        dim_y //= 2

        #########################
        # TODO: Add more layers #
        #########################

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels=8,
                                                 out_channels=16, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # input shape: in_channels x in x in = 32 x in x in
        # pooling: pool_size = 2, pool_stride = 2, pool_dilation = 1
        # output pooling shape: 32 x floor(in/2) x floor(in/2)
        # padding 1, stride 1, kernel 3 -> no change in dimensions
        # output conv shape: out_channels x floor(in/2) x floor(in/2)
        #                   =          24 x floor(in/2) x floor(in/2)

        dim_x //= 2
        dim_y //= 2

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(in_channels=16,
                                                 out_channels=32, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # input shape: in_channels x in x in = 32 x in x in
        # pooling: pool_size = 2, pool_stride = 2, pool_dilation = 1
        # output pooling shape: 32 x floor(in/2) x floor(in/2)
        # padding 1, stride 1, kernel 3 -> no change in dimensions
        # output conv shape: out_channels x floor(in/2) x floor(in/2)
        #                   =          24 x floor(in/2) x floor(in/2)

        dim_x //= 2
        dim_y //= 2

        self.conv5 = ai8x.MaxPool2d(kernel_size=2, stride=2)

        dim_x //= 2
        dim_y //= 2

        self.fc6 = ai8x.Linear(in_features=32 * dim_x * dim_y,
                               out_features=32, wide=True,
                               bias=True, **kwargs)
        self.fc7 = ai8x.Linear(in_features=32,
                               out_features=16, wide=True,
                               bias=True, **kwargs)
        self.fc8 = ai8x.Linear(in_features=16,
                               out_features=3, wide=True,
                               bias=True, **kwargs)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # # Data plotting - for debug
        # matplotlib.use('MacOSX')
        # plt.imshow(x[1, 0], cmap="gray")
        # plt.show()
        # breakpoint()
        
        x = self.conv1(x)
        # print(f"Post conv1 shape: {x.shape}")
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        # print(f"Post fcx shape: {x.shape}")

        # Loss chosed, CrossEntropyLoss, takes softmax into account already func.log_softmax(x, dim=1))

        return x


def afhqnet(pretrained=False, **kwargs):
    """
    Constructs a MemeNet model.
    """
    assert not pretrained
    return AFHQNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'afhqnet',
        'min_input': 1,
        'dim': 2,
    }
]

