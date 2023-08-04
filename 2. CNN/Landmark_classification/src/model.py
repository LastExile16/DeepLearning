import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # add same last layer 256

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.drop2d = nn.Dropout2d(p=dropout)
        self.drop2d_x2 = nn.Dropout2d(p=dropout*2)
        self.bn2d_1 = nn.BatchNorm2d(16)
        self.bn2d_2 = nn.BatchNorm2d(32)
        self.bn2d_3 = nn.BatchNorm2d(64)
        self.bn2d_4 = nn.BatchNorm2d(128)
        self.bn2d_5 = nn.BatchNorm2d(256)
        self.bn2d_6 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1d_1 = nn.BatchNorm1d(256)
        self.bn1d_2 = nn.BatchNorm1d(128)
        self.drop1d = nn.Dropout(p=dropout)
        self.drop1d_x2 = nn.Dropout(p=dropout*2)

        ### Sequential Method:


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        x = F.relu(self.maxpool(self.bn2d_1(self.conv1(x))))
        # x = self.drop2d(x)

        x = F.relu(self.maxpool(self.bn2d_2(self.conv2(x))))
        x = self.drop2d(x)

        x = F.relu(self.maxpool(self.bn2d_3(self.conv3(x))))
        x = self.drop2d(x)

        x = F.relu(self.maxpool(self.bn2d_4(self.conv4(x))))
        x = self.drop2d(x)

        x = F.relu(self.maxpool(self.bn2d_5(self.conv5(x))))
        x = self.drop2d(x)
        
        x = F.relu(self.maxpool(self.bn2d_6(self.conv6(x))))
        x = self.drop2d(x)
        
        # create a global average pooling here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1d_1(self.fc1(x)))
        x = self.drop1d(x)
        x = F.relu(self.bn1d_2(self.fc2(x)))
        x = self.drop1d(x)
        x = self.fc3(x)
        
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
