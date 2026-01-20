import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class MNISTNPYDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0 
        # image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  

        return image, label

def show_image(image):
    image = image.squeeze().numpy()
    plt.imshow(image, cmap='gray')
    plt.axis('off') 
    plt.show()


if __name__ == '__main__':
    train_images_path = '../data/MNIST/npy/train_images.npy'
    train_labels_path = '../data/MNIST/npy/train_labels.npy'

    dataset = MNISTNPYDataset(train_images_path, train_labels_path)

    for i in range(5):
        image, label = dataset[i]
        print(image.shape)
        print(label)
        # print(f"标签: {label}")
        # show_image(image) 
        break
