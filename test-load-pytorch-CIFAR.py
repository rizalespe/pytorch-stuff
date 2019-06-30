import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from CustomNet import CustomNet
import torch.nn as nn

data_dir = "/home/rizalespe/research/_datasets/cifar10"


# Define the transformation
transform = transforms.Compose([transforms.ToTensor()])

# Prepare the datasets
dataset = torchvision.datasets.CIFAR10(root=data_dir, transform=transform, train=True)

# Load the prepared dataset with certain settings
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the network, loss function, and optimization
net = CustomNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for id, data in enumerate(train_loader, 0):
    print(data[0].shape)
    input, label = data
    #print(label)

    # Forward section
    output = net(input)

    # Backward section



    # Optimize section
