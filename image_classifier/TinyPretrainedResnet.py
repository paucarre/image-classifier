import torchvision.models as models
import torch.autograd as autograd
import torch.nn as nn
import torch

class TinyPretrainedResnet(nn.Module):

    def __init__(self, number_of_classes):
        super(TinyPretrainedResnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.number_of_classes = number_of_classes
        self.fully_connected = nn.Conv2d(512, number_of_classes, 1, 1)
        torch.nn.init.xavier_uniform_(self.fully_connected.weight)
        self.fully_connected.bias.data.zero_()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        outputs = self.fully_connected(x)
        return outputs.squeeze()
