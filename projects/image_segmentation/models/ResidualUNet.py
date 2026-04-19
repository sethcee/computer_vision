import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualUNet(nn.Module) :

    def __init__(self, num_classes) :
        super(ResidualUNet, self).__init__()

        # set internal variables
        self.num_classes = num_classes

        # load pretrained Resnet18 Imagenet weights
        base_model = models.resnet18(weights = "DEFAULT")
        base_layers = list(base_model.children())

        # pre-trained encoder
        self.down_layer0 = nn.Sequential(*base_layers[0:4])
        self.down_layer1 = base_layers[4]
        self.down_layer2 = base_layers[5]
        self.down_layer3 = base_layers[6]
        self.down_layer4 = base_layers[7]
    
        # decoder
        # note:
        #        1. We will upsample in the forward function in order to dynamically set the spatial dimensions.
        #        2. The channel dimensions of the convolutional layers are set by the result of concatenating the output below
        #           and skip connection.

        # layer 1 up sample
        up_layer1_in_channels = self.down_layer4[1].conv2.out_channels + self.down_layer3[1].conv2.out_channels
        up_layer1_out_channels = self.down_layer3[1].conv2.out_channels
        self.up_layer1 = self._create_upsample_block(in_channels = up_layer1_in_channels, out_channels = up_layer1_out_channels)

        # layer 2 up sample
        up_layer2_in_channels = self.up_layer1[0].out_channels + self.down_layer2[1].conv2.out_channels
        up_layer2_out_channels = self.down_layer2[1].conv2.out_channels
        self.up_layer2 = self._create_upsample_block(in_channels = up_layer2_in_channels, out_channels = up_layer2_out_channels)

        # layer 3 up sample
        up_layer3_in_channels = self.up_layer2[0].out_channels + self.down_layer1[1].conv2.out_channels
        up_layer3_out_channels = self.down_layer1[1].conv2.out_channels
        self.up_layer3 = self._create_upsample_block(in_channels = up_layer3_in_channels, out_channels = up_layer3_out_channels)

        # decoder output layer
        decoder_in_channels = self.up_layer3[0].out_channels + self.down_layer0[0].out_channels
        decoder_out_channels = decoder_in_channels // 2
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels = decoder_in_channels, out_channels = decoder_out_channels, kernel_size = 3),
            nn.BatchNorm2d(decoder_out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels = decoder_out_channels, out_channels = self.num_classes, kernel_size = 3),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.num_classes, out_channels = self.num_classes, kernel_size = 1)
        )

    def _create_upsample_block(self,in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1) :
        upsample_layer = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return upsample_layer
    
    def _concatenate_inputs(self, smaller_input, bigger_input) :
        up_sampled_smaller_input = F.interpolate(smaller_input, size = bigger_input.shape[2:], mode = 'bilinear', align_corners = True)
        combined_input = torch.cat([up_sampled_smaller_input, bigger_input], dim = 1)
        return combined_input

    def forward(self, x) :
        
        # encode
        down_layer0_output = self.down_layer0(x)
        down_layer1_output = self.down_layer1(down_layer0_output)
        down_layer2_output = self.down_layer2(down_layer1_output)
        down_layer3_output = self.down_layer3(down_layer2_output)
        down_layer4_output = self.down_layer4(down_layer3_output)

        # decode
        up_layer1_input = self._concatenate_inputs(down_layer4_output, down_layer3_output)
        up_layer1_output = self.up_layer1(up_layer1_input)

        up_layer2_input = self._concatenate_inputs(up_layer1_output, down_layer2_output)
        up_layer2_output = self.up_layer2(up_layer2_input)

        up_layer3_input = self._concatenate_inputs(up_layer2_output, down_layer1_output)
        up_layer3_output = self.up_layer3(up_layer3_input)

        decoder_input = self._concatenate_inputs(up_layer3_output, down_layer0_output)
        logits_map = self.decoder(decoder_input)

        return logits_map