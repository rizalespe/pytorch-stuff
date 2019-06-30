from torch.utils.data import Dataset

class CustomDataset(Dataset):
    # load the dataset directory or files here
    def __init__(self):
        self.samples = list(range(1,1001))

    # return the length of the dataset
    def __len__(self):
        return len(self.samples)

    # return the item data
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # for inspecting and debugging purpose
    dataset = CustomDataset()
    print(len(dataset))
    print(dataset[100])
    print(dataset[1:10])
