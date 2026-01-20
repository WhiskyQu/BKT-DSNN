import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

class CIFAR100Dataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images = np.load(images_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(np.array(image)).float() / 255.0  
        image = image.permute(2, 0, 1)  # (batch_size, height, width, channels) -> (batch_size, channels, height, width)
        label = torch.tensor(label, dtype=torch.long)  

        return image, label


if __name__ == '__main__':
    train_images_path = '../data/CIFAR-100/npy/train_images.npy'
    train_labels_path = '../data/CIFAR-100/npy/train_labels.npy'

    dataset = CIFAR100Dataset(train_images_path, train_labels_path)

    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    for images, labels in data_loader:
        print(images.shape)
        # for i in range(len(images)):
        #     plt.imshow(images[i].numpy().astype(np.uint8))
        #     plt.title(f"Label: {labels[i].item()}") 
        #     plt.axis('off') 
        #     plt.show()
        break  
