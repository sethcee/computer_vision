import torch.nn as nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module) :
    """
    The residual block for a Residual Neural Network. 

    The ResidualBlock implements convolutional operations along with the identity/skip connection. Specifically,
    each block contains two sequential convolutional blocks (each of which, themselves, contain a 2d convolution
    operation, a batch norm, and ReLU) and a skip connection across the full length of these two blocks.

    Args :
        in_channels (type :int, required)  : Number of channels in the input tensor. 
        out_channels (type: int, required) : Number of channels produced by the convolution. i.e., the number of filters.
        stride (type: int, optional)       : The stride for the first convolutional block, used for downsampling. 
                                             Defaults to 1, in which case no
                                             spatial downsampling will occur in the residual block.
        downsample (type: nn.Module, opt)  : A module used to downsample the input so its dimensions match the output of the
                                             identity/skip connection. Defaults to None, in which case stride must be set to
                                             maintain the spatial dimensions. Note, padding = 1, and kernel_size = 3.
    """

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None) :
        super(ResidualBlock, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = stride,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(out_channels)
        )
        
        
        self.downsample = downsample

    def _initial_forward(self,x) :
        """
        Defines the main convolutional function of the residual block
        Args :
            x, of type torch.Tensor, the input tensor.
        Returns :
            out, of type Torch.Tensor, the output of the convolutional path.
        """
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        return out
        

    def forward(self,x) : 
        """
        Defines the forward path, which is the sum of the identity connection and the the convolutional path.
        Args :
            x, of type Torch.Tensor, the input tensor.
        Returns
            out, of type Torch.Tensor, the output of the residual block.
        """

        identity = x
        if self.downsample is not None :
            identity = self.downsample(identity)

        out = self._initial_forward(x)
        out += identity
        out = F.relu(out) #note F.relu() is a function while nn.ReLU() is a class; they perform the same operation
        return out
    


class ResidualNN(nn.Module) :

    """
    Args :
        num_classes (int, optional)     : The number of output classes for the final classification layer. 
                                          The default is 5.
        num_blocks (int list, optional) : A list defining the number of ResidualBlocks in the three main
                                          stages. The default is [2, 2, 2]
    """

    def __init__(self, num_classes = 10, num_blocks = [2, 2, 2], dropout_rate = 0.25) :
        super(ResidualNN, self).__init__()

        self.num_classes = num_classes
        self.in_channels = 32
        self.dropout_rate = dropout_rate
        
        # initial conv block
        self.initial_block = self._get_initial_block()

        # residual layers
        self.res_block1 = self._make_residual_block(32, num_blocks[0], stride = 1)
        self.res_block2 = self._make_residual_block(64, num_blocks[1], stride = 2)
        self.res_block3 = self._make_residual_block(128, num_blocks[2], stride = 2)

        # final classifier
        self.final_block = self._get_final_block()
    
    def _get_initial_block(self) :
        """
        Defines the initial convolutional layer
        """
        initial_block = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, bias = False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )
        return initial_block
    
    def _get_final_block(self) :
        """
        Defines the final classifier
        """
        final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),# adaptiveAvgPool2d lets us specify the target output size. It squashes 
                                        # the spatial dimensions down to (1,1) in this case letting us broadly
                                        # summarize the image. i.e., output tensor is (batch_size, num_channels, 1,1)
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.num_classes)

        )
        return final_block
    
    def _make_residual_block(self, out_channels, num_blocks, stride) :

        # downsample if needed
        downsample = None
        if stride != 1 or self.in_channels != out_channels :
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(out_channels)
            )

        # list to hold layers
        layers = []

        # get first block
        first_block = ResidualBlock(self.in_channels, out_channels, stride = stride, downsample = downsample)
        layers.append(first_block)

        # update number of in_channels
        self.in_channels = out_channels

        # the remaining blocks do not change spatial dimensions
        for _ in range(1, num_blocks) :
            block =ResidualBlock(self.in_channels, out_channels, stride = 1, downsample = None)
            layers.append(block)
        
        #
        return nn.Sequential(*layers)
    
    def forward(self,x) :

        # Final pass. Softmax Activation is bundled with CrossEntropyLoss in
        # PyTorch, which is why the forward pass returns logits.
        x = self.initial_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        logits = self.final_block(x)
        return logits

    