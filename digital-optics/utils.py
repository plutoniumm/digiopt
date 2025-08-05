import torchvision.transforms as transforms
from os.path import exists
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

def nam(path):
    return path.split("/")[-1].split(".")[0]


class Image2Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, root, transform=None):
        self.root = root
        self.paths = image_paths
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        if self.transform:
            img = self.transform(img)
        return img

    def file(self, index):
        return self.paths[index]

    def open(self, name):
        name = f"{self.root}/{name}.png"
        if name in self.cache:
            return self.cache[name]

        img = None
        if exists(name + ".png"):
            img = name + ".png"
        if exists(name + ".jpg"):
            img = name + ".jpg"

        img = Image.open(name)
        if self.transform:
            img = self.transform(img)

        self.cache[name] = img
        return img

    def __len__(self):
        return len(self.paths)


def Imageset(paths):
    root = paths
    if paths.endswith("/*"):
        root = paths[:-2]
    paths = glob(paths)
    paths.sort()

    return Image2Dataset(paths, root, transform=transform)
