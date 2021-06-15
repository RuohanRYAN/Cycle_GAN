import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self,MONET_FILENAMES,PHOTO_FILENAMES, transform=None):
        self.transform = transform
        self.PHOTO_FILENAMES = PHOTO_FILENAMES
        self.MONET_FILENAMES = MONET_FILENAMES
        if len(self.MONET_FILENAMES) > len (self.PHOTO_FILENAMES):
            self.MONET_FILENAMES, self.PHOTO_FILENAMES = self.PHOTO_FILENAMES, self.MONET_FILENAMES
        self.new_perm()

    def new_perm(self):
        self.randperm = torch.randperm(len(self.PHOTO_FILENAMES))[:len(self.MONET_FILENAMES)]
    def __getitem__(self, index):
        item_MONET_FILENAMES = self.transform(Image.open(self.MONET_FILENAMES[index % len(self.MONET_FILENAMES)]))
        item_PHOTO_FILENAMES = self.transform(Image.open(self.PHOTO_FILENAMES[self.randperm[index]]))
        if item_MONET_FILENAMES.shape[0] != 3:
            item_MONET_FILENAMES = item_MONET_FILENAMES.repeat(3, 1, 1)
        if item_PHOTO_FILENAMES.shape[0] != 3:
            item_PHOTO_FILENAMES = item_PHOTO_FILENAMES.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        return (item_MONET_FILENAMES - 0.5) * 2, (item_PHOTO_FILENAMES - 0.5) * 2

    def __len__(self):
        return min(len(self.MONET_FILENAMES), len(self.PHOTO_FILENAMES))