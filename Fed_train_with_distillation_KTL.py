import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
from tqdm import tqdm
from models.SNN import SNN_VGG9_BNTT, SNN_VGG11_BNTT
from models.SNN_DCLS import SNN_VGG9_BNTT_DCLS, SNN_VGG11_BNTT_DCLS
from models.SNN_GMBS import SNN_VGG9_BNTT_GMBS, SNN_VGG11_BNTT_GMBS
from Fed_dataset.MNIST import MNISTNPYDataset
from Fed_dataset.CIFAR10 import CIFAR10Dataset
from Fed_dataset.CIFAR100 import CIFAR100Dataset
from Fed_dataset.FashionMNIST import FashionMNISTNPYDataset
import random
import time

class ASFOptimizer:
    def __init__(self, params, base_lr=0.001, alpha=0.9, sparsity_threshold=0.1):
        self.params = list(params)
        self.base_lr = base_lr
        self.alpha = alpha
        self.sparsity_threshold = sparsity_threshold
        self.lr = [base_lr] * len(self.params)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()  

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                sparsity = torch.mean((param.data.abs() < self.sparsity_threshold).float())
                self.lr[i] = self.base_lr * (1.0 + sparsity.item()) ** self.alpha
                param.data -= self.lr[i] * param.grad.data


def save_model(model, save_path, accuracy, best_acc):
    if accuracy > best_acc:
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with accuracy: {accuracy:.4f}")
        return accuracy
    return best_acc

def federated_learning(model_class, trainloader, testloader, criterion, device, rounds, local_epochs,
                       client_selection_ratio, num_clients, model_args, save_path, save_frequency):
    client_models = [model_class(*model_args).to(device) for _ in range(num_clients)]
    # client_optimizers = [optim.Adam(client_model.parameters(), lr=0.001) for client_model in client_models]
    client_optimizers = [ASFOptimizer(client_model.parameters(), base_lr=0.001) for client_model in client_models]
    global_model = model_class(*model_args).to(device)  

    best_acc = 0.0 
    communication_rounds = 0  
    total_time_cost = 0.0 
    total_communication_cost = 0.0 

    def model_size(model):
        param_size = sum([param.numel() for param in model.parameters()])
        param_bits = param_size * 32  
        param_megabytes = param_bits / (8 * 1024 * 1024) 
        return param_megabytes

    for round in range(rounds):
        start_time = time.time()  

        selected_clients = random.sample(range(num_clients), int(num_clients * client_selection_ratio))
        print(f"Round {round + 1}/{rounds}, Selected Clients: {selected_clients}")

        for client_idx in selected_clients:
            print(f"Training client {client_idx} for {local_epochs} local epochs")

            train_model_with_ktl(client_models[client_idx], client_models, trainloader, criterion,
                                  client_optimizers[client_idx], device, local_epochs)

        global_state_dict = global_model.state_dict()
        for key in global_state_dict.keys():
            if random.random() < 0.5: 
                global_state_dict[key] = sum(client_models[i].state_dict()[key] for i in selected_clients) / len(
                    selected_clients)

        communication_rounds += len(selected_clients)

        communication_cost = model_size(client_models[0]) * len(selected_clients) 
        total_communication_cost += communication_cost

        for client_model in client_models:
            client_model.load_state_dict(global_state_dict)

        global_model.load_state_dict(global_state_dict)

        if (round + 1) % save_frequency == 0:
            torch.save(global_model.state_dict(), f"{save_path}_round{round + 1}.pth")
            print(f"Model saved at round {round + 1}")

        test_acc = evaluate_model(global_model, testloader, device)
        best_acc = save_model(global_model, f"{save_path}_best.pth", test_acc, best_acc)

        round_time = time.time() - start_time
        total_time_cost += round_time
        print(f"Round {round + 1} Time Cost: {round_time:.2f} seconds")

    torch.save(global_model.state_dict(), f"{save_path}_final.pth")
    print(f"Final model saved.")

    print(f"Total communication rounds: {communication_rounds}")
    print(f"Total time cost: {total_time_cost:.2f} seconds")
    print(f"Total communication cost: {total_communication_cost:.2f} MB")


def train_model_with_ktl(student_model, client_models, dataloader, criterion, optimizer, device, epochs):
    student_model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            student_outputs = student_model(images)

            ktl_loss = 0.0
            for model in client_models:
                model.eval()
                with torch.no_grad():
                    teacher_outputs = model(images)  
                ktl_loss += nn.MSELoss()(student_outputs, teacher_outputs)  

            loss = criterion(student_outputs, labels)
            total_loss = loss + ktl_loss 

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            _, predicted = torch.max(student_outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_loss = running_loss / len(dataloader)

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    acc = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description='Manage model training parameters with argparse.')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'fashion_mnist', 'mnist'],
                        default='cifar10', help='Name of the dataset to use')
    parser.add_argument('--image_size', type=int, choices=[32, 28], default=32,
                        help='Size of input images')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu',
                        help='Activation function to use')
    parser.add_argument('--loss_function', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy',
                        help='Loss function to use')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--timesteps', type=int, default=2, help='Number of time_window')
    parser.add_argument('--num_class', type=int, default=10, help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=3, help='channels of image')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='./saved_models', help='Path to save the model')
    parser.add_argument('--log_save_path', type=str, default='./training.log', help='Path to save the log')
    parser.add_argument('--save_frequency', type=int, default=5, help='Frequency of saving the model')
    parser.add_argument('--model_name', type=str, default='SNN_VGG9_BNTT', help='Model name for saving')

    parser.add_argument('--clients', type=int, default=5, help='Number of clients for federated learning')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs for each client')
    parser.add_argument('--client_selection_ratio', type=float, default=0.5, help='Client selection ratio for each round')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='prune_ratio')

    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    log_dir = os.path.dirname(args.log_save_path)
    if log_dir: 
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=args.log_save_path,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(f"Model:{args.model_name}")

    # transform = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    dataset_mapping = {
        'cifar10': CIFAR10Dataset,
        'cifar100': CIFAR100Dataset,
        'fashion_mnist': FashionMNISTNPYDataset,
        'mnist': MNISTNPYDataset
    }

    dataset_path_mapping = {
        'cifar10': ('../data/CIFAR-10/npy/train_images.npy', '../data/CIFAR-10/npy/train_labels.npy',
                    '../data/CIFAR-10/npy/test_images.npy', '../data/CIFAR-10/npy/test_labels.npy'),
        'cifar100': ('../data/CIFAR-100/npy/train_images.npy', '../data/CIFAR-100/npy/train_labels.npy',
                     '../data/CIFAR-100/npy/test_images.npy', '../data/CIFAR-100/npy/test_labels.npy'),
        'fashion_mnist': ('../data/FashionMNIST/npy/train_images.npy', '../data/FashionMNIST/npy/train_labels.npy',
                          '../data/FashionMNIST/npy/test_images.npy', '../data/FashionMNIST/npy/test_labels.npy'),
        'mnist': ('../data/MNIST/npy/train_images.npy', '../data/MNIST/npy/train_labels.npy', '../data/MNIST/npy/test_images.npy',
        '../data/MNIST/npy/test_labels.npy')
    }

    trainset = dataset_mapping[args.dataset](dataset_path_mapping[args.dataset][0], dataset_path_mapping[args.dataset][1], transform=None)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = dataset_mapping[args.dataset](dataset_path_mapping[args.dataset][2], dataset_path_mapping[args.dataset][3], transform=None)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    if args.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss()

    model_mapping = {
        'SNN_VGG9_BNTT': SNN_VGG9_BNTT,
        'SNN_VGG11_BNTT': SNN_VGG11_BNTT,
        'SNN_VGG9_BNTT_DCLS': SNN_VGG9_BNTT_DCLS,
        'SNN_VGG11_BNTT_DCLS': SNN_VGG11_BNTT_DCLS,
        'SNN_VGG9_BNTT_GMBS': SNN_VGG9_BNTT_GMBS,
        'SNN_VGG11_BNTT_GMBS': SNN_VGG11_BNTT_GMBS,
    }
    if args.model_name not in model_mapping:
        raise ValueError(f"Unknown model: {args.model_name}")

    if args.model_name in ['SNN_VGG9_BNTT', 'SNN_VGG11_BNTT',
                           'SNN_VGG9_BNTT_DCLS', 'SNN_VGG11_BNTT_DCLS']:
        model_args = [args.in_channels, args.timesteps, 0.95, args.image_size, args.num_class]
    elif args.model_name in ['SNN_VGG9_BNTT_GMBS', 'SNN_VGG11_BNTT_GMBS']:
        model_args = [args.in_channels, args.timesteps, 0.95, args.image_size, args.num_class, args.prune_ratio]
    elif args.model_name in ['SpikingConvNet', 'SNN_STBSTDP_DCLS']:
        model_args = [args.image_size, args.in_channels]
    elif args.model_name == 'ANN_VGG11':
        model_args = [args.in_channels, args.image_size, args.num_class]

    federated_learning(model_mapping[args.model_name], trainloader, testloader, criterion, device, args.rounds, args.local_epochs, args.client_selection_ratio, args.clients, model_args, args.model_save_path, args.save_frequency)


if __name__ == '__main__':
    main()
