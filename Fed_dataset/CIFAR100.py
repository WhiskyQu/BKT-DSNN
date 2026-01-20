import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image


class CIFAR100Dataset(Dataset):
    def __init__(self, images_file, labels_file, num_clients=5, add_noise=False, noise_level=0.1, transform=None, non_iid=True):
        self.images = np.load(images_file)
        self.labels = np.load(labels_file)
        self.transform = transform
        self.add_noise = add_noise
        self.noise_level = noise_level  

        if non_iid:
            self.data_indices = self.non_iid_partition(num_clients)
        else:
            self.data_indices = list(range(len(self.images))) 

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        image_idx = self.data_indices[idx]
        image = self.images[image_idx]
        label = self.labels[image_idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(np.array(image)).float() / 255.0  
        image = image.permute(2, 0, 1)  # (height, width, channels) -> (channels, height, width)

        if self.add_noise:
            noise = torch.randn(image.size()) * self.noise_level  
            image = image + noise
            image = torch.clamp(image, 0, 1)

        label = torch.tensor(label, dtype=torch.long)  

        return image, label

    def non_iid_partition(self, num_clients):
        class_indices = {i: np.where(self.labels == i)[0] for i in range(100)}

        client_data_indices = [[] for _ in range(num_clients)]

        for class_idx, indices in class_indices.items():
            np.random.shuffle(indices) 
            num_samples = len(indices)
            samples_per_client = num_samples // num_clients

            for client_idx in range(num_clients):
                start = client_idx * samples_per_client
                end = (client_idx + 1) * samples_per_client if client_idx < num_clients - 1 else num_samples
                client_data_indices[client_idx].extend(indices[start:end])

        return [idx for client_indices in client_data_indices for idx in client_indices]


if __name__ == '__main__':
    train_images_path = '../data/CIFAR-100/npy/train_images.npy'
    train_labels_path = '../data/CIFAR-100/npy/train_labels.npy'

    dataset = CIFAR100Dataset(train_images_path, train_labels_path, num_clients=5, add_noise=True, noise_level=0.1, non_iid=True)

    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    for images, labels in data_loader:
        print(images.shape)
        for i in range(len(images)):
            plt.imshow(images[i].permute(1, 2, 0).numpy())
            plt.title(f"Label: {labels[i].item()}")
            plt.axis('off')
            plt.show()
        break 
