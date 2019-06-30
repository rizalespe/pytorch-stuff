'''
    - sample code for testing the custom dataset class
    - test the function of DataLoader class
'''
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader

dataset = CustomDataset()
data = DataLoader(dataset, batch_size=10, shuffle=True)

for id, x in enumerate(data):
    print(x)
