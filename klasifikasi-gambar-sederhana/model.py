#Import needed packages
import torch
import torch.nn as nn
import torchvision.models as models


class CustomPretrainResNext50(nn.Module):
    def __init__(self, num_class):
        super(CustomPretrainResNext50, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Linear(model.fc.in_features, num_class)       
        self.model = model

    def forward(self, image):        
        out = self.model(image)
        return out

class CustomPretrainVGG16(nn.Module):
    def __init__(self, num_class):
        super(CustomPretrainVGG16, self).__init__()
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier[6] = nn.Linear(model.classifier[3].out_features, num_class)       
        self.model = model

    def forward(self, image):        
        out = self.model(image)
        return out

class CustomPretrainResnet18(nn.Module):
    def __init__(self, num_class):
        super(CustomPretrainResnet18, self).__init__()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Linear(model.fc.in_features, num_class)        
        self.model = model

    def forward(self, image):        
        out = self.model(image)
        return out

class CustomPretrainResnet152(nn.Module):
    def __init__(self, num_class):
        super(CustomPretrainResnet152, self).__init__()
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Linear(model.fc.in_features, num_class)        
        self.model = model

    def forward(self, image):        
        out = self.model(image)
        return out

class CustomPretrainAlexNet(nn.Module):
    def __init__(self, num_class):
        super(CustomPretrainAlexNet, self).__init__()
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    
        model.classifier[6] = nn.Linear(model.classifier[4].out_features, num_class)        
        self.model = model

    def forward(self, image):        
        out = self.model(image)
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
