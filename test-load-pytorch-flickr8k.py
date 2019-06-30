from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision

flickr_image_dir = "/home/rizalespe/research/_datasets/flickr8k/Flickr8k_Dataset"
ann_file = "/home/rizalespe/research/_datasets/flickr8k/Flickr8k_text/Flickr8k.token.txt"

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.Flickr8k(root=flickr_image_dir, ann_file=ann_file, transform=transform)
train_loader = DataLoader(dataset)

for i, (imgs, caps, caplens) in enumerate(train_loader):
    print(imgs, caps, caplens)
