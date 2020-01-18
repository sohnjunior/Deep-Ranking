from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class DatasetImageNet(Dataset):
    """ custumized dataset loader """

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index, :]
        images = [Image.open(row[i]).convert('RGB') for i in range(3)]  # open triplet images as RGB(query, neg, pos)

        if self.transform is not None:
            for i in range(0, 3):
                images[i] = self.transform(images[i])

        q_image, p_image, n_image = images[0], images[1], images[2]

        return q_image, p_image, n_image


# -- pre-processing component
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224, interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def euclidean_distance(x, y):
    """ calculate euclidean distance """
    return np.sqrt(np.sum(x - y, axis=1) ** 2)
