import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import torch

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class Image2Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)


def Imageset(paths):
    dataset = Image2Dataset(glob(paths), transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False), dataset
