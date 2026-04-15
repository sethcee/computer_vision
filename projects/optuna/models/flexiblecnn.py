import torch.nn as nn
import torch

class FlexibleCNN(nn.Module) :
    """
    A flexible Convolutional Neural Network that is defined by the provided hyperparameters.
    This flexible model is for use with Optuna, which automates the tuning process

    """

    def __init__(self, num_layers, num_filters, kernel_sizes, dropout_rate, fc_size) :
        """
        Initialization function for the FlexibleCNN class

        Args:
            num_layers : The number of convolutional blocks
            num_filters : A list of integers specifying the number of output filters for each
                          convolutional block.
            kernel-sizes : A list of integers specifying the kernel size for each convolutional layer.
            dropout_rate : The dropout probability in the classifier
            fc_size : The number of neurons in the fully connected layer
        """

        super(FlexibleCNN, self).__init__()

        # instantiate the feature feature extractor
        conv_blocks = []
        in_channels = 3
        for i in range(num_layers) :

            # get the layer parameters
            out_channels = num_filters[i]
            kernel_size = kernel_sizes[i]
            padding = (kernel_size - 1) // 2
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
            )
            conv_blocks.append(conv_block)
            in_channels = out_channels
            
        self.features = nn.Sequential(*conv_blocks)

        # the classifier will be set later in forward
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size
        self.classifier = None
        
    def _create_classifier(self, flattened_size, device) :

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(inplace = True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_size, 10)
        ).to(device)

    def forward(self,x) :
        """
        Defines the forward pass.

        Args :
            x : the input tensor with shape (batch_size, channels, height, width)
        Returns :
            Output logits of classifier.
        """
        # normally we wouldn't need the device here, but since we are dynamically creating 
        # the classifier we do need it. We can assume the input tensor, x, is on the 
        # correct device, however. The other layers would've been moved over when the
        # model was initialized.
        device = x.device

        # get features
        x = self.features(x)

        # flatten all but batch dimension and get the size
        x_flattened = torch.flatten(x,1)
        flattened_size = x_flattened.size(1)

        # create classifier if we haven't already
        if self.classifier is None :
            self._create_classifier(flattened_size, device)

        logits = self.classifier(x_flattened)
        return logits