import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from CustomNet import CustomNet
import torch.nn as nn
import torch
from torchsummary import summary

data_dir = "/home/rizalespe/research/_datasets/cifar10"

# Define the transformation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Prepare the datasets
dataset = torchvision.datasets.CIFAR10(root=data_dir, transform=transform, train=True)

# Load the prepared dataset with certain settings
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the network, loss function, and optimization
net = CustomNet()

summary(net, (3, 227, 227))
exit()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss=0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # Reset optimizer / zero the parameter of gradient
        optimizer.zero_grad()

        # Forward section
        outputs = net(inputs)

        # Backward section
        loss = criterion(outputs, labels)
        loss.backward()

        # Optimize section
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

torch.save(net.state_dict(), 'model.ckpt')
