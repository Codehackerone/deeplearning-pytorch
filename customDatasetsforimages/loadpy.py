import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class CatsAndDogsDataset(Dataset):
    # csv file->img filename and (0 or 1)
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)



## USe

dataset = CatsAndDogsDataset(csv_file='../dfdfdf.csv', root_dir="catsordogs",
                             transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
