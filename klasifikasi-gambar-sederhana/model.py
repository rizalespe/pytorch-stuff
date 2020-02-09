#Import needed packages
import torch
import torch.nn as nn
import torchvision.models as models

class CustomPretrain(nn.Module):
    def __init__(self, num_class):
        super(CustomPretrain, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        
        resnet.fc = nn.Linear(resnet.fc.in_features, num_class)        
        self.resnet = resnet

    def forward(self, image):        
        # image = image.view(-1, )
        out = self.resnet(image)
        return out

class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()

        self.layer1 = nn.Sequential(
                                        nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer2 = nn.Sequential(
                                        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        
        self.fc = nn.Linear(32*56*56, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
