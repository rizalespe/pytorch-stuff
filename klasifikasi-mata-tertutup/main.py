import argparse
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import CustomPretrainResNext50, CustomModel, CustomPretrainResnet18, CustomPretrainResnet152, CustomPretrainAlexNet, CustomPretrainVGG16
from torch.utils.tensorboard import SummaryWriter

def main(args):
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Log Tensorboard
    writer = SummaryWriter(args.tensorboard_dir)

    # Hyper parameters
    num_epochs = args.num_epoch
    num_classes = args.num_class
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    #Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.crop_size, padding=4),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #Define transformations for the test set
    test_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.crop_size, padding=4),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset configuration
    train_dataset = torchvision.datasets.ImageFolder(root=args.training_dir, 
                                                    transform=train_transformations)

    test_dataset = torchvision.datasets.ImageFolder(root=args.val_dir, 
                                                    transform=test_transformations)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    # Model initialization
    # model = CustomModel(num_classes).to(device)
    # model = CustomPretrainResnet18(num_classes).to(device)
    # model = CustomPretrainResnet152(num_classes).to(device)
    model = CustomPretrainAlexNet(num_classes).to(device)
    # model = CustomPretrainVGG16(num_classes).to(device)
    # model = CustomPretrainResNext50(num_classes).to(device)
    
    
    # Generate network diagrams
    images_for_graph, labels = next(iter(train_loader))
    images_for_graph = images_for_graph.to(device)
    writer.add_graph(model, images_for_graph)
    writer.close()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    all_step = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print(images.shape)
            # exit()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                test_loss = 0
                # Test the model
                model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        test_loss = criterion(outputs, labels)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    writer.add_scalars('loss', {'train': loss.item(), 'validation': test_loss.item()}, all_step)
                    writer.add_scalar('accuracy', 100 * correct / total, all_step)

                    #print('Test Accuracy of the model on the {} test images: {} %, test loss {}'.format(total, 100 * correct / total,test_loss))
                    print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Accuration: {} '.format(epoch+1, num_epochs, i+1, total_step, loss.item(), test_loss.item(), 100 * correct / total))
            all_step = all_step + 1

    # Save the model checkpoint
    torch.save(model.state_dict(), args.save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tensorboard_dir', type=str, default='', help = "merupakan direktori yang berisi log yang ditampilkan pada Tensorboard")
    parser.add_argument('--num_epoch', type=int, default=5, help = "jumlah iterasi/epoch yang dilakukan pada proses training")
    parser.add_argument('--num_class', type=int, default=4, help = "jumlah kelas yang ada pada dataset")
    parser.add_argument('--batch_size', type=int, default=1, help = "jumlah data yang terkandung pada setiap batch saat proses training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help = "nilai learning rate")
    parser.add_argument('--crop_size', type=int, default=32, help = "ukurang crop saat proses transformasi")
    parser.add_argument('--training_dir', type=str, help = "direktori yang berisi data untuk training")
    parser.add_argument('--val_dir', type=str, help = "direktori yang berisi data untuk validasi/testing")
    parser.add_argument('--save_model', type=str, default='model.ckpt', help = "lokasi file beserta nama file hasil model yang telah selesai dilakukan pelatihan")
    
    args = parser.parse_args()
    main(args)
